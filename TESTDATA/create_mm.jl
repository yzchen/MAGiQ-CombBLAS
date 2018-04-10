m, n, nnz = 10, 10, 10
if size(ARGS, 1) >= 3
  m = parse(Int64, ARGS[1])
  n = parse(Int64, ARGS[2])
  nnz = parse(Int64, ARGS[3])
else
  println("you should run this program like following : ")
  println("julia create_mm.jl 100 100 100")
  println("by default it will create a 10 * 10 with 10 non-zeros matrix")
end

x = rand(nnz, 2)
x[:, 1] *= m
x[:, 2] *= n
x += 1

x = trunc.(Int64, x)
x = vcat([m n], x)

y = rand(nnz, 1)
y *= 10

y = trunc.(Int64, y)
y = vcat(nnz, y)

triples = hcat(x, y)

writedlm("gen_$(m)_$(n)_$(nnz).txt", triples)
