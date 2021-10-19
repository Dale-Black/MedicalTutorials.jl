### A Pluto.jl notebook ###
# v0.16.0

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

# ╔═╡ 123c3f17-4a4a-478d-ae0e-5ec7f9d4ad44
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
end

# ╔═╡ d781f284-16c1-4505-9f5a-dbd90c0334d2
TableOfContents()

# ╔═╡ 6e2a1536-e659-4366-93de-ac7d2e7a27a4
md"""
## Load data
Part of the [Medical Decathlon Dataset](http://medicaldecathlon.com/)
"""

# ╔═╡ 855407b1-b8ce-465f-8531-1af5d014abe9
data_dir = raw"/Users/daleblack/Google Drive/Datasets/Task02_Heart"

# ╔═╡ 4cefe3ca-d7f1-4b2a-b63c-e39d734d2514
function loadfn_label(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{UInt8}, a)
    convert_a = convert_a .+ 1
    return convert_a
end

# ╔═╡ 838a17df-7479-42c2-a2c5-ce626a4016fe
function loadfn_image(p)
    a = NIfTI.niread(string(p)).raw

    convert_a = convert(Array{Float32}, a)
    convert_a = convert_a/max(convert_a...)
    return convert_a
end

# ╔═╡ 11d5a9b4-fdc3-48e4-a6ad-f7dec5fabb8b
begin
    niftidata_image(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
    niftidata_label(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
    data = (
        niftidata_image(joinpath(data_dir, "imagesTr")),
        niftidata_label(joinpath(data_dir, "labelsTr")),
    )
end

# ╔═╡ 349d843a-4a5f-44d7-9371-38c140b9972d
md"""
## Create learning method
"""

# ╔═╡ 778cc0f9-127c-4d4b-a7de-906dfcc29cae
train_files, val_files = MLDataPattern.splitobs(data, 0.8)

# ╔═╡ 019e666e-e2e4-4e0b-a225-b346c7c70939
struct ImageSegmentationSimple <: DLPipelines.LearningMethod
    imagesize
end

# ╔═╡ f7274fa9-8231-44fd-8d00-1c7ab7fc855c
image_size = (112, 112, 96)

# ╔═╡ 9ac18928-9fe4-46ed-ab9c-916791739157
method = ImageSegmentationSimple(image_size)

# ╔═╡ 854f8bc8-ebe3-4d65-bf5a-417b16ea94fb
md"""
### Set up `AddChannel` transform
"""

# ╔═╡ f461d63a-c5e6-4450-a80c-e15b6c2a56c0
struct MapItemData <: Transform
    f
end

# ╔═╡ ea9b74d8-89f0-4b02-b605-8d12c6becff0
begin
  DataAugmentation.apply(tfm::MapItemData, item::DataAugmentation.AbstractItem; randstate = nothing) = DataAugmentation.setdata(item, tfm.f(itemdata(item)))
  DataAugmentation.apply(tfm::MapItemData, item::DataAugmentation.Image; randstate = nothing) = DataAugmentation.setdata(item, tfm.f(itemdata(item)))
end

# ╔═╡ fe1123dd-7228-4fe4-8dca-d081cbbfda95
AddChannel() = MapItemData(a -> reshape(a, size(a)..., 1))

# ╔═╡ edf2b37a-2775-44c0-8d2e-5e2350b454c4
md"""
### Set up `encode` pipelines
"""

# ╔═╡ 7cf0cc6b-8ff9-4198-9646-9d0787e1013d
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
			AddChannel()
          )
      tfm_mask = OneHot()

      items = Image(Gray.(image)), MaskMulti(target)
      item_im, item_mask = apply(tfm_proj, (items))

      return apply(tfm_im, item_im), apply(AddChannel(), apply(tfm_mask, item_mask))
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
          AddChannel()
          )
      tfm_mask = OneHot()

      items = Image(Gray.(image)), MaskMulti(target)
      item_im, item_mask = apply(tfm_proj, (items))

      return apply(tfm_im, item_im), apply(AddChannel(), apply(tfm_mask, item_mask))
  end
end

# ╔═╡ f2a92ff7-0f94-44d9-aba7-4b7db9fa4a56
begin
	methoddata_train = DLPipelines.MethodDataset(train_files, method, Training())
	methoddata_valid = DLPipelines.MethodDataset(val_files, method, Validation())
end

# ╔═╡ b8b18728-cb5a-445a-809c-986e3965aad3
let
    x, y = MLDataPattern.getobs(methoddata_valid, 1)
    @assert size(x.data) == (image_size..., 1, 1)
    @assert size(y.data) == (image_size..., 2, 1)
end

# ╔═╡ bfe051e5-66a3-4b4b-b446-48195d8f3868
md"""
## Visualize
"""

# ╔═╡ 4b7bff28-ec60-4a9d-8b38-648ab871ed16
begin
    x, y = MLDataPattern.getobs(methoddata_valid, 3)
    x, y = x.data, y.data
end;

# ╔═╡ 08be9516-a44c-4fe5-ac46-f586600d586a
@bind b PlutoUI.Slider(1:size(x)[3], default=50, show_value=true)

# ╔═╡ e2670a05-2c09-4bd2-b40c-0cc46fef2344
heatmap(x[:, :, b, 1], colormap=:grays)

# ╔═╡ 427ff0c3-d773-4099-a483-bb8f7d4b8ba1
heatmap(y[:, :, b, 2], colormap=:grays)

# ╔═╡ 616aa780-cbcf-4bf2-b6c5-a984f2530482
md"""
## Create Dataloader
"""

# ╔═╡ bda7309e-ae97-4ac0-96b6-36a776e9215e
begin
    train_loader = DataLoaders.DataLoader(methoddata_train, 2)
    val_loader = DataLoaders.DataLoader(methoddata_valid, 2)
end

# ╔═╡ e70816b8-4597-4a54-b4f7-735880df6132
val_loader

# ╔═╡ 7e7a7905-57e8-4630-a5b5-30895b57d6b4
traindl, valdl = methoddataloaders(data, method)

# ╔═╡ 1cfae1c4-1531-4ed1-87df-6ca97e610015
# for (xs, ys) in traindl
# 	@show size(xs)
# 	@show size(ys)
# end

# ╔═╡ 70357c15-d0ee-41b2-a063-7e644e61ae94
# with_terminal() do
# 	for (xs, ys) in val_loader
# 	@show size(xs)
# 	@show size(ys)
# 	end
# end

# ╔═╡ 9d98ffe5-15b1-434d-bf92-bb35bd0ee831
# for (xs, ys) in val_loader
# 	@show size(xs)
# 	@show size(ys)
# end

# ╔═╡ 0c24a134-e25b-4e63-948f-0804e3fffa23
# for (xs, ys) in train_loader
# 	@assert size(xs.data) == (image_size..., 1, 4)
# 	@assert size(ys.data) == (image_size..., 2, 2)
# end

# ╔═╡ 4abe1a4a-65cd-4bed-bdae-ff31c69c5441
# begin
#   for epoch in 1:max_epochs
#       step = 0
#       @show epoch

#       # Loop through training data
#       for (xs, ys) in train_loader
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
#           for (val_xs, val_ys) in val_loader
#               val_step += 1
#               @show val_step

#               local val_ŷs = model(val_xs)
#               local val_loss = loss_function(val_ŷs[:, :, :, 2, :], val_ys[:, :, :, 2, :])

#               val_ŷs, val_ys = as_discrete(val_ŷs, 0.5), as_discrete(val_ys, 0.5)
#           end
#       end
#   end
# end

# ╔═╡ Cell order:
# ╠═123c3f17-4a4a-478d-ae0e-5ec7f9d4ad44
# ╠═d781f284-16c1-4505-9f5a-dbd90c0334d2
# ╟─6e2a1536-e659-4366-93de-ac7d2e7a27a4
# ╠═855407b1-b8ce-465f-8531-1af5d014abe9
# ╠═4cefe3ca-d7f1-4b2a-b63c-e39d734d2514
# ╠═838a17df-7479-42c2-a2c5-ce626a4016fe
# ╠═11d5a9b4-fdc3-48e4-a6ad-f7dec5fabb8b
# ╟─349d843a-4a5f-44d7-9371-38c140b9972d
# ╠═778cc0f9-127c-4d4b-a7de-906dfcc29cae
# ╠═019e666e-e2e4-4e0b-a225-b346c7c70939
# ╠═f7274fa9-8231-44fd-8d00-1c7ab7fc855c
# ╠═9ac18928-9fe4-46ed-ab9c-916791739157
# ╟─854f8bc8-ebe3-4d65-bf5a-417b16ea94fb
# ╠═f461d63a-c5e6-4450-a80c-e15b6c2a56c0
# ╠═ea9b74d8-89f0-4b02-b605-8d12c6becff0
# ╠═fe1123dd-7228-4fe4-8dca-d081cbbfda95
# ╟─edf2b37a-2775-44c0-8d2e-5e2350b454c4
# ╠═7cf0cc6b-8ff9-4198-9646-9d0787e1013d
# ╠═f2a92ff7-0f94-44d9-aba7-4b7db9fa4a56
# ╠═b8b18728-cb5a-445a-809c-986e3965aad3
# ╟─bfe051e5-66a3-4b4b-b446-48195d8f3868
# ╠═4b7bff28-ec60-4a9d-8b38-648ab871ed16
# ╟─08be9516-a44c-4fe5-ac46-f586600d586a
# ╠═e2670a05-2c09-4bd2-b40c-0cc46fef2344
# ╠═427ff0c3-d773-4099-a483-bb8f7d4b8ba1
# ╟─616aa780-cbcf-4bf2-b6c5-a984f2530482
# ╠═bda7309e-ae97-4ac0-96b6-36a776e9215e
# ╠═e70816b8-4597-4a54-b4f7-735880df6132
# ╠═7e7a7905-57e8-4630-a5b5-30895b57d6b4
# ╠═1cfae1c4-1531-4ed1-87df-6ca97e610015
# ╠═70357c15-d0ee-41b2-a063-7e644e61ae94
# ╠═9d98ffe5-15b1-434d-bf92-bb35bd0ee831
# ╠═0c24a134-e25b-4e63-948f-0804e3fffa23
# ╠═4abe1a4a-65cd-4bed-bdae-ff31c69c5441
