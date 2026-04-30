using Documenter
using POPS

DocMeta.setdocmeta!(POPS, :DocTestSetup, :(using POPS); recursive=true)

makedocs(;
    modules=[POPS],
    authors="Noe Blassel",
    sitename="POPS.jl",
    format=Documenter.HTML(;
        canonical="https://noeblassel.github.io/POPS.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/noeblassel/POPS.jl.git",
    devbranch="main",
)
