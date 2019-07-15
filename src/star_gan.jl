using Flux, Flux.Data.MNIST, Statistics
using Flux: Tracker, throttle, params, binarycrossentropy, crossentropy
using Flux.Tracker: update!
using Flux:@treelike
using NNlib: relu, leakyrelu
using Base.Iterators: partition
using Images: channelview
using CSV
using BSON: @save
using CuArrays
using CUDAnative:exp, log

include("data_loader.jl")
BATCH_SIZE = 512
training_set_size = 512 * 395
train_data = load_dataset_as_batches("/home/thebhatman/Downloads/celeba-dataset/img_align_celeba/", BATCH_SIZE)
train_data = gpu.(train_data)
attr_file = CSV.File("/home/thebhatman/Downloads/celeba-dataset/list_attr_celeba.csv")
labels = Array{Array{Float64, 1}, 1}(undef, training_set_size)
i = 0
for row in attr_file
  global i += 1
  labels[i] = ([row.Black_Hair, row.Blond_Hair, row.Brown_Hair, row.Male, row.Young] .+ 1)/2
  # push!(labels, ([row.Black_Hair, row.Blond_Hair, row.Brown_Hair, row.Male, row.Young] .+ 1)/2)
  if i == training_set_size
    break
  end
end
batched_labels = []
for x in partition(labels, BATCH_SIZE)
  push!(batched_labels, cat(x..., dims = 2))
end

λ = 1.0f0
γ = 10.0f0
struct ResidualBlock
  conv_layers
  norm_layers
  shortcut
end

@treelike ResidualBlock

function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, strides::Array{Tuple{Int,Int}}, shortcut = identity)
  local conv_layers = Array{Any, 1}(undef, length(filters) - 1)
  local norm_layers = Array{Any, 1}(undef, length(filters) - 1)
  for i in 2:length(filters)
    conv_layers[i - 1] = Conv(kernels[i-1], filters[i-1]=>filters[i], pad = pads[i-1], stride = strides[i-1])
    norm_layers[i - 1] = InstanceNorm(filters[i])
  end
  ResidualBlock(Tuple(conv_layers),Tuple(norm_layers),shortcut)
end

function ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut = identity)
  ResidualBlock(filters, [(i,i) for i in kernels], [(i,i) for i in pads], [(i,i) for i in strides], shortcut)
end

function (block::ResidualBlock)(input)
  local value = copy.(input)
  for i in 1:length(block.conv_layers)-1
    value = relu.((block.norm_layers[i])((block.conv_layers[i])(value)))
  end
  relu.(((block.norm_layers[end])((block.conv_layers[end])(value))) + block.shortcut(input))
end

generator = Chain(Conv((7, 7), 3 => 64, stride = (1, 1), pad = (3, 3)),
                InstanceNorm(64, relu),
                Conv((4, 4), 64=>128, stride = (2, 2), pad = (1, 1)),
                InstanceNorm(128, relu),
                Conv((4, 4), 128=>256, stride = (2, 2), pad = (1, 1)),
                InstanceNorm(256, relu),
                ResidualBlock([256, 256, 256], [(3, 3), (3, 3)], [(1, 1), (1, 1)], [(1, 1), (1, 1)]),
                ResidualBlock([256, 256, 256], [(3, 3), (3, 3)], [(1, 1), (1, 1)], [(1, 1), (1, 1)]),
                ResidualBlock([256, 256, 256], [(3, 3), (3, 3)], [(1, 1), (1, 1)], [(1, 1), (1, 1)]),
                ResidualBlock([256, 256, 256], [(3, 3), (3, 3)], [(1, 1), (1, 1)], [(1, 1), (1, 1)]),
                ResidualBlock([256, 256, 256], [(3, 3), (3, 3)], [(1, 1), (1, 1)], [(1, 1), (1, 1)]),
                ResidualBlock([256, 256, 256], [(3, 3), (3, 3)], [(1, 1), (1, 1)], [(1, 1), (1, 1)]),
                ConvTranspose((4, 4), 256=>128, stride = (2, 2), pad = (1,  1)),
                InstanceNorm(128),
                ConvTranspose((4, 4), 128=>64, stride = (2, 2), pad = (1, 1)),
                InstanceNorm(64),
                Conv((7, 7), 64=>3, stride = (1, 1), pad = (3, 3)),
                x -> tanh.(x)) |> gpu

discriminator = Chain(Conv((4, 4), 3=>64, stride = (2, 2), pad = (1, 1)),
                    x -> leakyrelu.(x, 0.01),
                    Conv((4, 4), 64=>128, stride = (2, 2), pad = (1, 1)),
                    x -> leakyrelu.(x, 0.01),
                    Conv((4, 4), 128=>256, stride = (2, 2), pad = (1, 1)),
                    x -> leakyrelu.(x, 0.01),
                    Conv((4, 4), 256=>512, stride = (2, 2), pad = (1, 1)),
                    x -> leakyrelu.(x, 0.01),
                    Conv((4, 4), 512=>1024, stride = (2, 2), pad = (1, 1)),
                    x -> leakyrelu.(x, 0.01),
                    Conv((4, 4), 1024=>2048, stride = (2, 2), pad = (1, 1)),
                    x -> leakyrelu.(x, 0.01)) |> gpu

discriminator_logit = Chain(discriminator, Conv((2, 2), 2048=>1, stride = (1, 1))) |> gpu

discriminator_classifier = Chain(discriminator, Conv((2, 2), 2048=>5, stride = (1, 1)),
                                    x -> reshape(x, 5, :)) |> gpu

function gan_loss(X, Y)
  logit = discriminator_logit(X)
  return mean(logit .* (-Y) .+ log.(ones(size(logit)...) .+ exp.(logit)))
end

function cls_loss(X, Y)
  class_probs = discriminator_classifier(X)
  return mean(class_probs .*(-Y) .+ log.(ones(size(class_probs)...) .+ exp.(class_probs)))
end

function recon_loss(real_img, fake_img)
  return mean(abs.(real_img - fake_img))
end

function discriminator_loss(X, target_class)
  d_real_adv_loss = gan_loss(X, ones(1, 1, 1, BATCH_SIZE))
  d_fake_adv_loss = gan_loss(generator(X), zeros(1, 1, 1, BATCH_SIZE))
  d_adv_loss = d_real_adv_loss + d_fake_adv_loss
  d_real_cls_loss = cls_loss(X, target_class)
  return d_adv_loss + λ * d_real_cls_loss
end

function generator_loss(X, target_class)
  fake_img = generator(X)
  g_adv_loss = gan_loss(fake_img, ones(1, 1, 1, BATCH_SIZE))
  g_fake_cls_loss = cls_loss(fake_img, target_class)
  g_recon_loss = recon_loss(X, fake_img)
  return g_adv_loss + λ * g_fake_cls_loss + γ * g_recon_loss
end

opt_disc = ADAM()
opt_gen = ADAM()

function training(X)
  target_labels = rand([1 0 0 1 1], 5, BATCH_SIZE)
  disc_grads = Tracker.gradient(()->discriminator_loss(X[1], X[2]), params(params(discriminator_logit)..., params(discriminator_classifier)...))
  Flux.Optimise.update!(opt_disc, params(params(discriminator_logit)..., params(discriminator_classifier)...), disc_grads)

  gen_grads = Tracker.gradient(()->generator_loss(X[1], target_labels), params(generator))
  Flux.Optimise.update!(opt_gen, params(generator), gen_grads)
  
  return discriminator_loss(X, target_labels), generator_loss(X, target_labels)
end

function save_weights(discriminator_logit, discriminator_classifier, generator)
  discriminator_logit = discriminator_logit |> cpu
  discriminator_classifier = discriminator_classifier |> cpu
  generator = generator |> cpu
  @save "../weights/discriminator_logit.bson" discriminator_logit
  @save "../weights/discriminator_classifier.bson" discriminator_classifier
  @save "../weights/generator.bson" generator
  discriminator_logit = discriminator_logit |> gpu
  discriminator_classifier = discriminator_classifier |> gpu
  generator = generator |> gpu
end

NUM_EPOCHS = 50

function train()
  i = 0
  for epoch in 1:NUM_EPOCHS
    println("---------------EPOCH : $epoch----------------")
    for d in zip(train_data, batched_labels)
      disc_loss, generator_loss = training(d |> gpu)
      println("Discriminator loss : $disc_loss, Generator loss : $generator_loss")
      i += 1
      if i % 1000 == 0
        save_weights(discriminator_logit, discriminator_classifier, generator)
      end
    end
  end
end




