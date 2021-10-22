### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ a7fc3207-40eb-4ca0-af05-5f6c09f80fd4
begin
    let
        using Pkg
        Pkg.activate(mktempdir())
        Pkg.Registry.update()
        Pkg.add("PlutoUI")
        Pkg.add("Tar")
        Pkg.add("MLDataPattern")
        Pkg.add("Glob")
        Pkg.add("NIfTI")
        Pkg.add("DataAugmentation")
        Pkg.add("CairoMakie")
        Pkg.add("ImageCore")
        Pkg.add("DataLoaders")
        # Pkg.add("CUDA")
        Pkg.add("FastAI")
        Pkg.add("StaticArrays")
		# Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
    end

    using PlutoUI
    using Tar
    using MLDataPattern
    using Glob
    using NIfTI
    using DataAugmentation
    using DataAugmentation: OneHot, Image
    using CairoMakie
    using ImageCore
    using DataLoaders
    # using CUDA
    using FastAI
    using StaticArrays
	# using DistanceTransforms
end

# ╔═╡ 55b595e1-b289-4e9e-a631-ccdb4fb790f0
TableOfContents()

# ╔═╡ 352addad-5193-4769-ad30-be73119e927e
md"""
## Load data
Part of the [Medical Decathlon Dataset](http://medicaldecathlon.com/)
"""

# ╔═╡ a1506bb2-7b2f-4c24-9ef9-6a6b09ac2360
data_dir = raw"/Users/daleblack/Google Drive/Datasets/Task02_Heart"

# ╔═╡ 957ca346-209d-4152-a5b1-48f234d1b390
function loadfn_label(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{UInt8}, a)
    convert_a = convert_a .+ 1
    return convert_a
end

# ╔═╡ 7e2ee61d-4c7c-4237-97e2-4606f5c52192
function loadfn_image(p)
    a = NIfTI.niread(string(p)).raw

    convert_a = convert(Array{Float32}, a)
    convert_a = convert_a/max(convert_a...)
    return convert_a
end

# ╔═╡ ab7fb11a-e94e-48ab-8d25-e76591dd52a7
begin
    niftidata_image(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
    niftidata_label(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
    data = (
        niftidata_image(joinpath(data_dir, "imagesTr")),
        niftidata_label(joinpath(data_dir, "labelsTr")),
    )
end

# ╔═╡ ebf8306c-6c80-47c6-bac9-5a773981282a
md"""
## BlockMethod
Mimic [link](https://fluxml.ai/FastAI.jl/dev/notebooks/imagesegmentation.ipynb.html)
"""

# ╔═╡ 9c02bdc5-d46e-49f0-8848-cdcc3f77bff7
testmethod = BlockMethod(
    (FastAI.Image{3}(), Mask{3}(1:2)),
    (
        ProjectiveTransforms((112, 112, 96)),
        ImagePreprocessing(; C=Gray{N0f8}, means=SVector(1), stds=SVector(1)),
        FastAI.OneHot()
    )
)

# ╔═╡ c70b66c7-2f01-4bbd-9a20-3723f7d8168f
md"""
## Check Method
"""

# ╔═╡ 492f96d5-ffd5-427e-bae6-ea09626b450b
begin
    image, mask = sample = getobs(data, 1);
    checkblock(testmethod.blocks, sample)
end

# ╔═╡ 70170b0b-6320-428b-8ef0-64a09249a54c
describemethod(testmethod)

# ╔═╡ 23742f4f-bace-4575-8ce2-906c0cef3898
md"""
## Create model
"""

# ╔═╡ 2cec3af0-dbb6-4cb7-8050-c56425605a00
begin
    # 3D layer utilities
    conv = (stride, in, out) -> Flux.Conv((3, 3, 3), in=>out, stride=stride, pad=Flux.SamePad())
    tran = (stride, in, out) -> Flux.ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=Flux.SamePad())

    conv1 = (in, out) -> Flux.Chain(conv(1, in, out), Flux.BatchNorm(out), x -> leakyrelu.(x))
    conv2 = (in, out) -> Flux.Chain(conv(2, in, out), Flux.BatchNorm(out), x -> leakyrelu.(x))
    tran2 = (in, out) -> Flux.Chain(tran(2, in, out), Flux.BatchNorm(out), x -> leakyrelu.(x))
end

# ╔═╡ 1986f393-36ed-499b-b73a-eed65f36f876
begin
    function unet3D(in_chs, lbl_chs)
        # Contracting layers
        l1 = Flux.Chain(conv1(in_chs, 4))
        l2 = Flux.Chain(l1, conv1(4, 4), conv2(4, 16))
        l3 = Flux.Chain(l2, conv1(16, 16), conv2(16, 32))
        l4 = Flux.Chain(l3, conv1(32, 32), conv2(32, 64))
        l5 = Flux.Chain(l4, conv1(64, 64), conv2(64, 128))

        # Expanding layers
        l6 = Flux.Chain(l5, tran2(128, 64), conv1(64, 64))
        l7 = Flux.Chain(Flux.Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))
        l8 = Flux.Chain(Flux.Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))
        l9 = Flux.Chain(Flux.Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))
        l10 = Flux.Chain(l9, conv1(4, lbl_chs))
    end
end

# ╔═╡ a5b998de-a93a-4f87-8569-b3da7f07642d
model = unet3D(1, 2)

# ╔═╡ e2b6a66f-aaf3-4673-99ca-7af37abd677c
md"""
## Helper functions
"""

# ╔═╡ d0cb59f8-26f6-4aab-b799-55d81d3e57b2
function dice_metric(ŷ, y)
    dice = 2 * sum(ŷ .& y) / (sum(ŷ) + sum(y))
    return dice
end

# ╔═╡ 47f8d85c-7ab3-4d54-ad42-6280cacc76e0
function as_discrete(array, logit_threshold)
    array = array .>= logit_threshold
    return array
end

# ╔═╡ 431015e9-3269-492c-9d7c-a95bc3414152
md"""
## Loss functions
"""

# ╔═╡ bdaef8df-8a85-489b-8a1e-1e527d7aede5
function dice_loss(ŷ, y)
    ϵ = 1e-5
    return loss = 1 - ((2 * sum(ŷ .* y) + ϵ) / (sum(ŷ .* ŷ) + sum(y .* y) + ϵ))
end

# ╔═╡ 0eb81afb-a055-42f4-97c3-eddd8fa1258e
function hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    M = (ŷ .- y) .^ 2 .* (ŷ_dtm .^ 2 .+ y_dtm .^ 2)
    return loss = mean(M)
end

# ╔═╡ 321387d9-59b8-452d-a44c-8b4b9211d79b
md"""
## Training
"""

# ╔═╡ 610c5d25-8ad8-47ee-a530-b76640261e9d
ps = Flux.params(model);

# ╔═╡ 97799e4a-fd24-4615-a763-72b7384a8c25
loss_function = Flux.Losses.dice_coeff_loss

# ╔═╡ 40be8808-40f2-4e87-b040-a554f0e2e10f
optimizer = Flux.ADAM(0.01)

# ╔═╡ 55e0f48e-90f6-4007-837b-d93ee138fa66
traindl, validdl = methoddataloaders(data, testmethod, 2)

# ╔═╡ 4997a1eb-6c50-463e-bf3e-e359634db323
learner = Learner(model, (traindl, validdl), optimizer, loss_function)

# ╔═╡ 61f55c98-c10f-41e1-9a73-ef4b31187a55
# fitonecycle!(learner, 1, 0.033)

# ╔═╡ d5e633c5-8913-4795-827a-2e6ffaf3549f


# ╔═╡ 99f2c227-e681-41bf-b165-b92ae9664de6
# begin
#   for (xs, ys) in traindl
#       @assert size(xs) == (112, 112, 96, 1, 2)
#       @assert size(ys) == (112, 112, 96, 2, 2)
#   end
# end

# ╔═╡ acc87036-0e03-4a2a-9bc7-241f3c8491e1
# begin
#   for (xs, ys) in validdl
#       @assert size(xs) == (112, 112, 96, 1, 4)
#       @assert size(ys) == (112, 112, 96, 2, 4)
#   end
# end

# ╔═╡ bf3b5453-4a86-4849-9937-1f2e6dadab65
# with_terminal() do
#   for (xs, ys) in validdl
#   @show size(ys)
#   end
# end

# ╔═╡ 3ba4ba5a-894b-4ab3-a43f-60ec8fb9823b
# begin
#   max_epochs = 2
#   val_interval = 1
#   epoch_loss_values = []
#   val_epoch_loss_values = []
#   dice_metric_values = []
# end

# ╔═╡ 5345afc0-a149-4daa-bda9-7c36d49b2efc
# begin
#   for epoch in 1:max_epochs
#       step = 0
#       @show epoch

#       # Loop through training data
#       for (xs, ys) in traindl
#           step += 1
#           @show step

#           gs = Flux.gradient(ps) do
#               ŷs = model(xs)
#               loss = loss_function(ŷs[:, :, :, 2, :], ys[:, :, :, 2, :])
#               return loss
#           end
#           Flux.update!(optimizer, ps, gs)
#       end

#       # Loop through validation data
#       if (epoch + 1) % val_interval == 0
#           val_step = 0
#           for (val_xs, val_ys) in validdl
#               val_step += 1
#               @show val_step

#               local val_ŷs = model(val_xs)
#               local val_loss = loss_function(val_ŷs[:, :, :, 2, :], val_ys[:, :, :, 2, :])

#               val_ŷs, val_ys = as_discrete(val_ŷs, 0.5), as_discrete(val_ys, 0.5)
#           end
#       end
#   end
# end

# ╔═╡ c999e14b-cc35-4b00-9e91-ce7d5644f118
# for epoch in 1:max_epochs
#     epoch_loss = 0
#     step = 0
#     @show epoch

#     # Loop through training data
#     for (xs, ys) in train_loader
#         step += 1
#         @show step

#         xs, ys = xs |> gpu, ys |> gpu

#         ŷs = model(xs)

#         outputs_soft = softmax(ŷs; dims = 2)
#         loss_seg_dice = dice_loss(outputs_soft[:, 2, :, :, :], ys== 1)

#         gt_dtm = compute_dtm(ys)
#         seg_dtm = compute_dtm(outputs_soft[:, 2, :, :, :] .> 0.5)
        
#         gs = Flux.gradient(ps) do
#             loss_hd = hd_loss(outputs_soft, seg_dtm, gt_dtm)
#             loss = alpha*(loss_seg_dice) + (1 - alpha) * loss_hd
#             return loss
#         end
#         Flux.update!(optimizer, ps, gs)
#         epoch_loss += loss_function(ŷs[:, :, :, 2, :], ys[:, :, :, 2, :])

#         println("$(step/(len(train_ds) // train_loader.batchsize))")
#         println("train_loss: $(loss:.4f)")
        
#     end
    
#     epoch_loss = (epoch_loss / step)
#     push!(epoch_loss_values, epoch_loss)

#     println("epoch $(epoch_num + 1) average loss: (epoch_loss:.4f)")
#     alpha -= 0.001
#     if alpha <= 0.001:
#         alpha = 0.001
#     end

#     # Loop through validation data
#     if (epoch + 1) % val_interval == 0
#         val_step = 0
#         val_epoch_loss = 0
#         metric_step = 0
#         dice = 0
#         for (val_xs, val_ys) in val_loader
#             val_step += 1
#             @show val_step

#             val_xs, val_ys = val_xs |> gpu, val_ys |> gpu
#             val_ŷs = model(val_xs)

#             outputs_soft = softmax(val_ŷs; dims = 2)
#             loss_seg_dice = dice_loss(outputs_soft[:, 2, :, :, :], ys== 1)

#             gt_dtm = compute_dtm(val_ys)
#             seg_dtm = compute_dtm(outputs_soft[:, 2, :, :, :] .> 0.5)
        
#             loss_hd = hd_loss(outputs_soft, seg_dtm, gt_dtm)
#             val_loss = alpha*(loss_seg_dice) + (1 - alpha) * loss_hd

#             val_epoch_loss += val_loss

#             val_ŷs, val_ys = val_ŷs |> cpu, val_ys |> cpu
#             metric_step += 1
#             metric = dice_metric(val_ŷs[:, :, :, 2, :], val_ys[:, :, :, 2, :])
#             dice += metric
#         end

#         val_epoch_loss = (val_epoch_loss / val_step)
#         push!(val_epoch_loss_values, val_epoch_loss)

#         dice = dice / metric_step
#         push!(metric_values, dice)

#     end  
# end

# ╔═╡ Cell order:
# ╠═a7fc3207-40eb-4ca0-af05-5f6c09f80fd4
# ╠═55b595e1-b289-4e9e-a631-ccdb4fb790f0
# ╟─352addad-5193-4769-ad30-be73119e927e
# ╠═a1506bb2-7b2f-4c24-9ef9-6a6b09ac2360
# ╠═957ca346-209d-4152-a5b1-48f234d1b390
# ╠═7e2ee61d-4c7c-4237-97e2-4606f5c52192
# ╠═ab7fb11a-e94e-48ab-8d25-e76591dd52a7
# ╟─ebf8306c-6c80-47c6-bac9-5a773981282a
# ╠═9c02bdc5-d46e-49f0-8848-cdcc3f77bff7
# ╟─c70b66c7-2f01-4bbd-9a20-3723f7d8168f
# ╠═492f96d5-ffd5-427e-bae6-ea09626b450b
# ╠═70170b0b-6320-428b-8ef0-64a09249a54c
# ╟─23742f4f-bace-4575-8ce2-906c0cef3898
# ╠═2cec3af0-dbb6-4cb7-8050-c56425605a00
# ╠═1986f393-36ed-499b-b73a-eed65f36f876
# ╠═a5b998de-a93a-4f87-8569-b3da7f07642d
# ╟─e2b6a66f-aaf3-4673-99ca-7af37abd677c
# ╠═d0cb59f8-26f6-4aab-b799-55d81d3e57b2
# ╠═47f8d85c-7ab3-4d54-ad42-6280cacc76e0
# ╟─431015e9-3269-492c-9d7c-a95bc3414152
# ╠═bdaef8df-8a85-489b-8a1e-1e527d7aede5
# ╠═0eb81afb-a055-42f4-97c3-eddd8fa1258e
# ╟─321387d9-59b8-452d-a44c-8b4b9211d79b
# ╠═610c5d25-8ad8-47ee-a530-b76640261e9d
# ╠═97799e4a-fd24-4615-a763-72b7384a8c25
# ╠═40be8808-40f2-4e87-b040-a554f0e2e10f
# ╠═55e0f48e-90f6-4007-837b-d93ee138fa66
# ╠═4997a1eb-6c50-463e-bf3e-e359634db323
# ╠═61f55c98-c10f-41e1-9a73-ef4b31187a55
# ╠═d5e633c5-8913-4795-827a-2e6ffaf3549f
# ╠═99f2c227-e681-41bf-b165-b92ae9664de6
# ╠═acc87036-0e03-4a2a-9bc7-241f3c8491e1
# ╠═bf3b5453-4a86-4849-9937-1f2e6dadab65
# ╠═3ba4ba5a-894b-4ab3-a43f-60ec8fb9823b
# ╠═5345afc0-a149-4daa-bda9-7c36d49b2efc
# ╠═c999e14b-cc35-4b00-9e91-ce7d5644f118
