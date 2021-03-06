using Knet
using FileIO
#################################PrepareData#####################
function underSampler(x,y)
    trueIndexList = find(y)
    falseIndexList = find(.!y)

    if length(falseIndexList) > length(trueIndexList)
        shuffledIndexes = randperm(length(falseIndexList))
        subSuffledIndexes = shuffledIndexes[1:length(trueIndexList)]
        falseIndexList = falseIndexList[subSuffledIndexes]
    elseif length(trueIndexList) > length(falseIndexList)
        shuffledIndexes = randperm(length(trueIndexList))
        subSuffledIndexes = shuffledIndexes[1:length(falseIndexList)]
        trueIndexList = trueIndexList[subSuffledIndexes]
    end

    indexList = sort(vcat(trueIndexList,falseIndexList))
    return x[indexList,:],y[indexList,:]
end

function generateWindowedData(x,y,window_size)
    sample_size = length(x[:,1])

    #Adding Window with window size 2*window_shift+1
    window_shift = Int((window_size - 1) / 2)

    windows = []
    for n = -window_shift:window_shift
        push!(windows, x[(window_shift+1+n):(sample_size-window_shift+n),:])
    end

    x = hcat(windows...)
    y = y[window_shift+1:sample_size-window_shift]

    return x, y
end

function prepareData(fileprefix,window_size)
    ##Class 1---Without NeuroD2, Class 2---With NeuroD2
    data = load("levent.jld2","data")

    y=data[:,1]
    x=data[:,2:end]

    x,y = generateWindowedData(x,y,window_size)

    #Undersampling the Data to obtain balanced DataSet
    x,y = underSampler(x,y)

    chr_data =  transpose(x)
    y_data = transpose(y)

    y_data = Array{Int64}(y_data)
    y_data = 1 + vec(y_data)

    #Creating Shuffled IndexList
    num_samples = length(chr_data[1,:])
    num_trn_samples = Int(floor(num_samples * 0.9))
    num_val_samples = Int(floor((num_samples - num_trn_samples - 2) / 2))
    num_test_samples = num_val_samples
    indexes = randperm(num_samples)
    trn_index_list = indexes[1:num_trn_samples]
    val_index_list = indexes[num_trn_samples+1:num_trn_samples+num_val_samples]
    test_index_list = indexes[num_val_samples+num_trn_samples+1:num_test_samples+num_val_samples+num_trn_samples]

    chr_trn = chr_data[:,trn_index_list]
    y_trn = y_data[trn_index_list]
    chr_val = chr_data[:,val_index_list]
    y_val = y_data[val_index_list]
    chr_test = chr_data[:,test_index_list]
    y_test = y_data[test_index_list]

    return chr_trn,y_trn,chr_test,y_test
end
####################################################################

#############################ExecuteData############################
function executeData(fileprefix,window_size,batchsize)
    xtrn,ytrn,xtst,ytst = prepareData(fileprefix,window_size)
    global dtrn = minibatch(xtrn, ytrn, batchsize;shuffle=true,xtype=KnetArray{Float32})
    global dtst = minibatch(xtst, ytst, batchsize;shuffle=true,xtype=KnetArray{Float32})

    num_rows = length(xtrn[:,1])
    return num_rows
end
####################################################################

#############################ML_Functions###########################
function sl()
    println("******************************************************************")
end

#prediction for both neural network models and regression models
function predict(w,x)
    for i=1:2:length(w)
        x = w[i]*mat(x) .+ w[i+1]
        if i<length(w)-1
            x = relu.(x)
        end
    end
    return x
end

loss(w,x,y_actual) = nll(predict(w,x),y_actual)
lossgradient = grad(loss)

function Optimization_Algorithm_Initializer(lr,length_of_w)
    Optimization_Algorithm = Sgd(lr = lr)
    m = Any[]
    for n=1:length_of_w
        push!(m,Optimization_Algorithm)
    end
    return m
end

function train(w, dtrn; lr=.01, epochs=10)
    optimization = Optimization_Algorithm_Initializer(lr,length(w))

    #global trn_loss = Any[]

    print_weight("Initial Values Are:", w)
    println("Starting the epoch 1:")
    for epoch=1:epochs
        for (x,y) in dtrn
            #push!(trn_loss,loss(w,x,y))
            g = lossgradient(w, x, y)
            update!(w,g,optimization)
        end

        if (epoch % 3) == 1
            println("epoch ",epoch," is over")
        end

    end
    print_weight("Final Values Are:", w)
    return w
end

#initialize weights for both neural network models and regression models
function initialize_weight(args...)
    weight = Any[]

    for i=2:length(args)
        push!(weight, xavier(args[i],args[i-1]))
        push!(weight, zeros(args[i],1))
    end
    return map(KnetArray{Float32}, weight)
end

function print_weight(message, weight)
    weight = map(Array{Float32}, weight)
    println(message)
    for i=1:2:length(weight)
        sl()
        println("Weights Values")
        println(weight[i])
        println("Bias Values")
        println(weight[i+1])
        sl()
    end
end
####################################################################

#############################Main################################
fileprefix = "chr1"
batchsize = 100

window_size_list = [1,3,5,7,9,11,13,15,17,19,25,31,39,45,51,55,61,67]
trn_accuracy_values = Any[]
tst_accuracy_values = Any[]
for window_size in window_size_list
    num_parameters = executeData(fileprefix,window_size,batchsize)
    w = initialize_weight(num_parameters,64,2)
    w = train(w, dtrn; lr=.01, epochs=10)
    push!(trn_accuracy_values, accuracy(w,dtrn,predict))
    push!(tst_accuracy_values, accuracy(w,dtst,predict))
    println("Training and Test Accuracy Values Are Calculated for the Window Size:",window_size)
end

trn_accuracy_values
tst_accuracy_values
#################################################################
#trn_loss
