# Run by the describe_output tool (describe_output.py), which replaces __PATH__
# with the path as a Julia string literal.
let path = __PATH__
    import Jutul
    files = isdir(path) ? sort(filter(f -> endswith(f, ".jld2"), readdir(path; join = true))) : [path]
    isempty(files) && error("no .jld2 files at $path")
    io = IOBuffer()
    fmt(x) = x isa AbstractFloat ? string(round(x; sigdigits = 4)) : string(x)
    function walk(x, name, depth)
        pad = "  "^depth
        if x isa AbstractDict
            ks = sort(collect(keys(x)); by = string)
            vs = [x[k] for k in ks]
            # A collection of like records (wells, say): name them, show one.
            if length(ks) > 3 && all(v -> v isa AbstractDict && keys(v) == keys(vs[1]), vs)
                println(io, pad, name, ": ", length(ks), " entries with the same fields (", join(string.(ks), ", "), "); the first:")
                walk(vs[1], string(ks[1]), depth + 1)
                return
            end
            if length(ks) > 30
                println(io, pad, name, ": Dict with ", length(ks), " keys, e.g. ", join(string.(ks[1:8]), ", "))
                return
            end
            println(io, pad, name, ":")
            foreach(k -> walk(x[k], string(k), depth + 1), ks)
        elseif x isa AbstractArray{<:Real} && !isempty(x)
            lo, hi = extrema(x)
            println(io, pad, name, ": ", eltype(x), " array ", join(size(x), "x"), ", range ", fmt(lo), " .. ", fmt(hi))
        elseif x isa AbstractArray
            println(io, pad, name, ": ", eltype(x), " array ", join(size(x), "x"))
        elseif x isa AbstractString
            println(io, pad, name, ": \"", first(x, 160), length(x) > 160 ? "..." : "", "\"")
        else
            println(io, pad, name, ": ", fmt(x))
        end
    end
    d = Jutul.JLD2.load(first(files))
    if length(files) > 1
        println(io, length(files), " files: ", join(basename.(files), ", "))
        metas = [get(Jutul.JLD2.load(f), "meta", Dict()) for f in files]
        varying = [k for k in keys(metas[1]) if length(unique(get.(metas, k, nothing))) > 1 && k != "created_utc"]
        for k in sort(varying)
            println(io, "meta.", k, " per file: ", join(fmt.(get.(metas, k, nothing)), ", "))
        end
        println(io, "Structure of ", basename(first(files)), " (the others match):")
    end
    if haskey(d, "report_time_s")
        t = d["report_time_s"]
        println(io, "time grid: ", length(t), " saved steps, end times ", fmt(t[1] / 86400), " .. ", fmt(t[end] / 86400), " days")
    end
    foreach(k -> walk(d[k], k, 0), sort(collect(keys(d))))
    print(String(take!(io)))
end
