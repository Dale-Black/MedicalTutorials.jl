using MedicalTutorials
using Documenter

makedocs(;
    modules=[MedicalTutorials],
    authors="Dale <djblack@uci.edu> and contributors",
    repo="https://github.com/Dale-Black/MedicalTutorials.jl/blob/{commit}{path}#L{line}",
    sitename="MedicalTutorials.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Dale-Black.github.io/MedicalTutorials.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Dale-Black/MedicalTutorials.jl",
)
