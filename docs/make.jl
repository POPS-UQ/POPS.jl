using Documenter
using POPSmodels

DocMeta.setdocmeta!(POPSmodels, :DocTestSetup, :(using POPSmodels); recursive=true)

makedocs(;
    modules=[POPSmodels],
    authors="POPS-UQ organization",
    sitename="POPSmodels.jl",
    format=Documenter.HTML(;
        canonical="https://POPS-UQ.github.io/POPSmodels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API reference" => "api.md",
        "Fitting an ACE model" => "ace.md",
        "Quantifying uncertainties in MD" => "md.md",
    ],
)

deploydocs(;
    repo="github.com/POPS-UQ/POPS.jl.git",
    devbranch="main",
)