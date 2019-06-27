using Base.Iterators: partition
using Images

function img_load(img_name)
    img = load(img_name)
    img = imresize(img, 128, 128)
    img = reshape(Float64.(channelview(img)), 128, 128, 3)
    return img
end

function load_dataset_as_batches(path, BATCH_SIZE)
    data = []
    i = 0
    for r in readdir(path)
        img_path = string(path, r)
        push!(data, img_load(img_path))
        i+=1
        if i == 4096
            break
        end
    end
    num_images = length(data)
    #println(num_images)
    batched_data = []
    for x in partition(data, BATCH_SIZE)
        x = reshape(cat(x..., dims = 4), 128, 128, 3, BATCH_SIZE)
        push!(batched_data, x)
    end
    return batched_data
end
