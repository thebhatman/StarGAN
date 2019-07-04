using Flux, Flux.Data.MNIST, Statistics
using Flux: Tracker, throttle, params, binarycrossentropy, crossentropy
using Flux.Tracker: update!
using Flux:@treelike
using NNlib: relu, leakyrelu
using Base.Iterators: partition
using Images: channelview

include("data_loader.jl")
BATCH_SIZE = 512
train_data = load_dataset_as_batches("C:/Users/manju/Downloads/celeba-dataset/img_align_celeba/img_align_celeba/", BATCH_SIZE)
train_data = gpu.(train_data)

λ = 1
γ = 10
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
                x -> tanh.(x))

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
                    x -> leakyrelu.(x, 0.01))

discriminator_logit = Chain(discriminator, Conv((2, 2), 2048=>1, stride = (1, 1)))

discriminator_classifier = Chain(discriminator, Conv((2, 2), 2048=>5, stride = (1, 1)),
                                    x -> reshape(x, 5, :))

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
  d_real_adv_loss = gan_loss(X, ones(1, 1, 1, 1))
  d_fake_adv_loss = gan_loss(generator(X), zeros(1, 1, 1, 1))
  d_adv_loss = d_real_adv_loss + d_fake_adv_loss
  d_real_cls_loss = cls_loss(X, target_class)
  return d_adv_loss + λ * d_real_cls_loss
end

function generator_loss(X, target_class)
  fake_img = generator(X)
  g_adv_loss = gan_loss(fake_img, ones(1, 1, 1, 1))
  g_fake_cls_loss = cls_loss(fake_img, target_class)
  g_recon_loss = recon_loss(X, fake_img)
  return g_adv_loss + λ * g_fake_cls_loss + γ * g_recon_loss
end
