#ML functions module

function sl()
    println("******************************************************************")
end

#Defining the fuctions
predict(w,x) = Knet.sigm(x*w)

function loss(w,x,y,batch_range,batch_size)
    x_converted = KnetArray{Float32}(x[batch_range,:])
    y_converted = KnetArray{Float32}(y[batch_range,1:1])
    result = -(1. / batch_size) * (transpose(y_converted) * KnetArray{Float32}(log.(predict(w,x_converted))) + (1 .- transpose(y_converted)) * KnetArray{Float32}(log.(1 .- predict(w,x_converted))))
    return result
end

function lossgradient(w,x,y,batch_range,batch_size)
    x_converted = KnetArray{Float32}(x[batch_range,:])
    y_converted = KnetArray{Float32}(y[batch_range,1:1])
    return (1. / batch_size) .* (transpose(x_converted) * (predict(w,x_converted) - y_converted))
end

#regular mini-batching, please use parser.java to import ranges for oversampled mini-batches
function mini_batch_rangeFinder(x,y,n,num_samples)
    #omitting the last chunk
    size = Int(floor(num_samples / n))
    last_batch_size = num_samples % n
    array_of_ranges = Array{UnitRange{Int64}}(size)
    starting_point = 1
    ending_point = n
    for nm = 1:size
        array_of_ranges[nm] = starting_point:ending_point
        starting_point += n
        ending_point += n
    end
    return array_of_ranges
end

function print_weight(message, w)
    println(message)
    for i=1:16
        println(w[i,1])
    end
    sl()
end
