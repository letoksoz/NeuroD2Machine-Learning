#READING THE DATA CREATED BY JAVA
fileprefix = "chr1"

#Creating a BitArray with the right dimensions
num_rows = countlines(string(fileprefix , ".data"))
line = readline(string(fileprefix , ".data"))
modifiedline = chomp(line)
num_columns = endof(modifiedline)

chr1array = falses(num_rows,num_columns)

#Check whether there is a newline or not
isnumber(modifiedline)

#Reading the DATA
lines = readlines(string(fileprefix , ".data"))

for l = 1:num_rows
   for i = 1:num_columns

       if lines[l][i] == '1'
           chr1array[l,i] = true
       end

   end
end

#WRITING THE DATA AS BITARRAY FOR FASTER PROCESSING IN JULIA
writedlm(string(fileprefix , "num_rows.txt"), num_rows)
writedlm(string(fileprefix , "num_columns.txt"), num_columns)
write(string(fileprefix , "bitarray.txt"), chr1array)
