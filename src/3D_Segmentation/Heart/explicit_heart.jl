### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 514a8cf8-15ff-4896-8905-3afcaa8b4d3a
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
		Pkg.add("DLPipelines")
		Pkg.add("DataAugmentation")
		Pkg.add("CairoMakie")
		Pkg.add("ImageCore")
		Pkg.add("DataLoaders")
		Pkg.add("Flux")
		Pkg.add("CUDA")
		Pkg.add(url="https://github.com/FluxML/FastAI.jl")
	end
	
	using PlutoUI
	using Tar
	using MLDataPattern
	using Glob
	using NIfTI
	using DLPipelines
	using DataAugmentation
	using DataAugmentation: OneHot, Image
	using CairoMakie
	using ImageCore
	using DataLoaders
	using Flux
	using CUDA
	using FastAI
end

# ╔═╡ 48f904bc-af7c-46b0-84a6-f0f45fcffd7d
TableOfContents()

# ╔═╡ 5f77cbb8-26a4-4287-b4a3-c9c1158f8d1f
md"""
## Load data
"""

# ╔═╡ 7ec89eaf-7eca-47d9-90fc-c6b9fa241f2e
data_dir = "/Users/daleblack/Google Drive/Datasets/Task02_Heart"

# ╔═╡ 4294429e-e29e-4986-b0d7-66d8a1d7d3a5
function loadfn_label(p)
	a = NIfTI.niread(string(p)).raw
	convert_a = convert(Array{UInt8}, a)
	return convert_a
end

# ╔═╡ 3dc4d917-357f-4bf5-8554-e4ddf08bdc2c
function loadfn_image(p)
	a = NIfTI.niread(string(p)).raw
	convert_a = convert(Array{Float32}, a)
	return convert_a
end

# ╔═╡ ae9ad741-39f8-4991-bd90-9516f3f9ba10
begin
	
	niftidata_image(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
	niftidata_label(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
	data = (
		niftidata_image(joinpath(data_dir, "imagesTr")),
		niftidata_label(joinpath(data_dir, "labelsTr")),
	)
	train_files, val_files = splitobs(data, 0.75)
end

# ╔═╡ 6a7bc2b9-786f-4d8f-b546-883961fbe36e
md"""
## Create learning method
"""

# ╔═╡ b19eddb1-22c3-42b7-ab57-ecca033e0198
struct ImageSegmentationSimple <: DLPipelines.LearningMethod
    imagesize
end

# ╔═╡ 1e984c9f-0d87-45db-9a12-c182d1f47856
image_size = (112, 112, 96)

# ╔═╡ 11fc9a9a-f0e4-4df9-a5c4-2e6404dbd86f
method = ImageSegmentationSimple(image_size)

# ╔═╡ a3e23cd9-476f-49f5-8477-a4ce6c7fc7cb
md"""
### Set up `AddChannel` transform
"""

# ╔═╡ a514108d-e739-4c12-8d42-7a900bc23987
struct MapItemData <: Transform
    f
end

# ╔═╡ ae250c91-99c0-41aa-9445-f3b24eac6dcf
begin
	DataAugmentation.apply(tfm::MapItemData, item::DataAugmentation.AbstractItem; randstate = nothing) = DataAugmentation.setdata(item, tfm.f(itemdata(item)))
	DataAugmentation.apply(tfm::MapItemData, item::DataAugmentation.Image; randstate = nothing) = DataAugmentation.setdata(item, tfm.f(itemdata(item)))
end

# ╔═╡ a4fe5e5a-3e6a-4a73-9d4f-39fcde1aa7f8
AddChannel() = MapItemData(a -> reshape(a, size(a)..., 1))

# ╔═╡ 57e35785-3639-4f68-b993-3cecd26369a4
md"""
### Set up `encode` pipelines
"""

# ╔═╡ bd717212-ccf8-41f3-9da2-059a5129008b
begin
	function DLPipelines.encode(
			method::ImageSegmentationSimple,
			context::Training,
			(image, target)::Union{Tuple, NamedTuple}
			)
		
		tfm_proj = RandomResizeCrop(method.imagesize)
		tfm_im = DataAugmentation.compose(
			ImageToTensor(),
			NormalizeIntensity(),
			# AddChannel()
			)
		tfm_mask = OneHot()
		
		items = Image(ImageCore.colorview(Gray, image)), MaskMulti(target .+ 1)
		item_im, item_mask = apply(tfm_proj, (items))
		
		return apply(tfm_im, item_im), apply(tfm_mask, item_mask)
	end

	function DLPipelines.encode(
			method::ImageSegmentationSimple,
			context::Validation,
			(image, target)::Union{Tuple, NamedTuple}
			)
		
		tfm_proj = CenterResizeCrop(method.imagesize)
		tfm_im = DataAugmentation.compose(
			ImageToTensor(),
			NormalizeIntensity(),
			# AddChannel()
			)
		tfm_mask = OneHot()
		
		items = Image(ImageCore.colorview(Gray, image)), MaskMulti(target .+ 1)
		item_im, item_mask = apply(tfm_proj, (items))
		
		return apply(tfm_im, item_im), apply(tfm_mask, item_mask)
	end
end

# ╔═╡ f4a24f2f-2b97-4d19-a221-bea10fc70752
begin
	methoddata_train = DLPipelines.MethodDataset(train_files, method, Training())
	methoddata_valid = DLPipelines.MethodDataset(val_files, method, Validation())
end

# ╔═╡ ac774864-f414-48b2-a166-a5bdd722b199
md"""
## Plot data
"""

# ╔═╡ 46a375aa-b645-4cd6-9363-eeec2a6ef740
begin
	x, y = MLDataPattern.getobs(methoddata_train, 3)
	x, y = x.data, y.data
end;

# ╔═╡ 0f819e4f-538c-4269-b851-6cc43d08fdf2
@bind a PlutoUI.Slider(1:size(x)[3], default=50, show_value=true)

# ╔═╡ e3b91342-550f-4762-828e-9e832b0a43f1
heatmap(x[:, :, a, 1], colormap=:grays)

# ╔═╡ 4e5fe0d9-389e-4f4a-9838-ac909ae28416
heatmap(y[:, :, a, 2], colormap=:grays)

# ╔═╡ 14a23b5f-8cbf-47d5-a428-995ddbf9d239
md"""
## Create dataloader
"""

# ╔═╡ 5449c0d1-22f4-44b3-a7f5-09a7829aebf2
begin
	train_loader = DataLoaders.DataLoader(methoddata_train, 4)
	val_loader = DataLoaders.DataLoader(methoddata_valid, 2)
end

# ╔═╡ af3a74ab-3bec-466e-9d00-2c19f092654d
# for (xs, ys) in train_loader
# 	@assert size(xs) == (image_size..., 1, 4)
# 	@assert size(ys) == (image_size..., 2, 2)
# end

# ╔═╡ 608f919d-a5e4-4782-8620-2a2079955989
md"""
## Create  model
"""

# ╔═╡ f3d86bf5-ea8a-4c86-957d-6cbfbd5727a2
begin
	# 3D layer utilities
	conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=SamePad())
	tran = (stride, in, out) -> ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=SamePad())

	conv1 = (in, out) -> Chain(conv(1, in, out), BatchNorm(out), x -> leakyrelu.(x))
	conv2 = (in, out) -> Chain(conv(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
	tran2 = (in, out) -> Chain(tran(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
end

# ╔═╡ 239ec2b2-1948-4ecf-a082-e25d8686b33d
begin
	function unet3D(in_chs, lbl_chs)
		# Contracting layers
		l1 = Chain(conv1(in_chs, 4))
		l2 = Chain(l1, conv1(4, 4), conv2(4, 16))
		l3 = Chain(l2, conv1(16, 16), conv2(16, 32))
		l4 = Chain(l3, conv1(32, 32), conv2(32, 64))
		l5 = Chain(l4, conv1(64, 64), conv2(64, 128))

		# Expanding layers
		l6 = Chain(l5, tran2(128, 64), conv1(64, 64))
		l7 = Chain(Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))
		l8 = Chain(Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))
		l9 = Chain(Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))
		l10 = Chain(l9, conv1(4, lbl_chs))
	end
end

# ╔═╡ cb0b4939-3dcf-4a51-866d-47653e346e70
md"""
#### Helper functions
"""

# ╔═╡ b654aa5b-0e4a-4271-a092-11adce871ffd
function dice_metric(ŷ, y)
    dice = 2 * sum(ŷ .& y) / (sum(ŷ) + sum(y))
    return dice
end

# ╔═╡ da2d5337-566a-45a5-a881-704acd3625c8
function as_discrete(array, logit_threshold)
    array = array .>= logit_threshold
    return array
end

# ╔═╡ fc0b4656-eb18-4284-9cd6-4a3510df818b
md"""
## Train model
"""

# ╔═╡ dd338b37-6036-4958-9feb-f926b9e1599f
begin
	model = unet3D(1, 2) |> gpu
	ps = Flux.params(model)
	loss_function = Flux.Losses.dice_coeff_loss
	optimizer = Flux.ADAM(0.01)
end

# ╔═╡ 2f69eaea-9a54-411e-84bc-466cc3a90738
begin
	max_epochs = 30
	val_interval = 2
	epoch_loss_values = []
	val_epoch_loss_values = []
	dice_metric_values = []
end

# ╔═╡ 764ebf3a-1172-420b-96dc-3bdf645e8f3d
# for epoch in 1:max_epochs
#     epoch_loss = 0
#     step = 0
#     println("Epoch:", epoch)

#     # Loop through training data
#     for (xs, ys) in train_loader
#         step += 1
#         println("train step: ", step)

#         xs, ys = xs |> gpu, ys |> gpu
#         gs = Flux.gradient(ps) do
#             ŷs = model(xs)
#             loss = loss_function(ŷs[:, :, :, 2, :], ys[:, :, :, 2, :])
#             return loss
#         end
#         Flux.update!(optimizer, ps, gs)

#         local ŷs = model(xs)
#         local loss = loss_function(ŷs[:, :, :, 2, :], ys[:, :, :, 2, :])
#         epoch_loss += loss
#     end
#     epoch_loss = (epoch_loss / step)
#     push!(epoch_loss_values, epoch_loss)

#     # Loop through validation data
#     if (epoch + 1) % val_interval == 0
#         val_step = 0
#         val_epoch_loss = 0
#         metric_step = 0
#         dice = 0
#         for (val_xs, val_ys) in val_loader
#             val_step += 1
#             println("val step: ", val_step)

#             val_xs, val_ys = val_xs |> gpu, val_ys |> gpu
#             local val_ŷs = model(val_xs)
#             local val_loss = loss_function(val_ŷs[:, :, :, 2, :], val_ys[:, :, :, 2, :])
#             val_epoch_loss += val_loss

#             val_ŷs, val_ys = val_ŷs |> cpu, val_ys |> cpu
#             val_ŷs, val_ys = as_discrete(val_ŷs, 0.5), as_discrete(val_ys, 0.5)
#             metric_step += 1
#             metric = dice_metric(val_ŷs[:, :, :, 2, :], val_ys[:, :, :, 2, :])
#             dice += metric
#         end

#         val_epoch_loss = (val_epoch_loss / val_step)
#         push!(val_epoch_loss_values, val_epoch_loss)

#         dice = dice / metric_step
#         push!(dice_metric_values, dice)
#     end
# end

# ╔═╡ d1305a5e-201f-453c-a82a-629d82c7f5a8
md"""
## Create dataloaders
"""

# ╔═╡ 66dd57fb-f03e-4d72-ab4a-8aa2d2e6c397
batchsize = 2

# ╔═╡ 28f2a867-b1e2-4003-8255-5ca21cb421f8
traindl, validdl = methoddataloaders(data, method, batchsize)

# ╔═╡ be19b6b9-5ecf-4e6f-87aa-9a49ac4552e8
md"""
## Create learner
"""

# ╔═╡ 2c3a0eb8-fe32-4e59-a2c7-7da268207869
learner = Learner(model, (traindl, validdl), optimizer, loss_function)

# ╔═╡ df9b1c9d-c6c0-459f-9ac0-e6e84eb47cae
md"""
## Train
"""

# ╔═╡ 387c737e-5478-4668-ad6e-562f8caeeb49
# fitonecycle!(learner, 1)

# ╔═╡ Cell order:
# ╠═514a8cf8-15ff-4896-8905-3afcaa8b4d3a
# ╠═48f904bc-af7c-46b0-84a6-f0f45fcffd7d
# ╟─5f77cbb8-26a4-4287-b4a3-c9c1158f8d1f
# ╠═7ec89eaf-7eca-47d9-90fc-c6b9fa241f2e
# ╠═4294429e-e29e-4986-b0d7-66d8a1d7d3a5
# ╠═3dc4d917-357f-4bf5-8554-e4ddf08bdc2c
# ╠═ae9ad741-39f8-4991-bd90-9516f3f9ba10
# ╟─6a7bc2b9-786f-4d8f-b546-883961fbe36e
# ╠═b19eddb1-22c3-42b7-ab57-ecca033e0198
# ╠═1e984c9f-0d87-45db-9a12-c182d1f47856
# ╠═11fc9a9a-f0e4-4df9-a5c4-2e6404dbd86f
# ╟─a3e23cd9-476f-49f5-8477-a4ce6c7fc7cb
# ╠═a514108d-e739-4c12-8d42-7a900bc23987
# ╠═ae250c91-99c0-41aa-9445-f3b24eac6dcf
# ╠═a4fe5e5a-3e6a-4a73-9d4f-39fcde1aa7f8
# ╟─57e35785-3639-4f68-b993-3cecd26369a4
# ╠═bd717212-ccf8-41f3-9da2-059a5129008b
# ╠═f4a24f2f-2b97-4d19-a221-bea10fc70752
# ╟─ac774864-f414-48b2-a166-a5bdd722b199
# ╠═46a375aa-b645-4cd6-9363-eeec2a6ef740
# ╟─0f819e4f-538c-4269-b851-6cc43d08fdf2
# ╠═e3b91342-550f-4762-828e-9e832b0a43f1
# ╠═4e5fe0d9-389e-4f4a-9838-ac909ae28416
# ╟─14a23b5f-8cbf-47d5-a428-995ddbf9d239
# ╠═5449c0d1-22f4-44b3-a7f5-09a7829aebf2
# ╠═af3a74ab-3bec-466e-9d00-2c19f092654d
# ╟─608f919d-a5e4-4782-8620-2a2079955989
# ╠═f3d86bf5-ea8a-4c86-957d-6cbfbd5727a2
# ╠═239ec2b2-1948-4ecf-a082-e25d8686b33d
# ╟─cb0b4939-3dcf-4a51-866d-47653e346e70
# ╠═b654aa5b-0e4a-4271-a092-11adce871ffd
# ╠═da2d5337-566a-45a5-a881-704acd3625c8
# ╟─fc0b4656-eb18-4284-9cd6-4a3510df818b
# ╠═dd338b37-6036-4958-9feb-f926b9e1599f
# ╠═2f69eaea-9a54-411e-84bc-466cc3a90738
# ╠═764ebf3a-1172-420b-96dc-3bdf645e8f3d
# ╟─d1305a5e-201f-453c-a82a-629d82c7f5a8
# ╠═66dd57fb-f03e-4d72-ab4a-8aa2d2e6c397
# ╠═28f2a867-b1e2-4003-8255-5ca21cb421f8
# ╟─be19b6b9-5ecf-4e6f-87aa-9a49ac4552e8
# ╠═2c3a0eb8-fe32-4e59-a2c7-7da268207869
# ╟─df9b1c9d-c6c0-459f-9ac0-e6e84eb47cae
# ╠═387c737e-5478-4668-ad6e-562f8caeeb49
