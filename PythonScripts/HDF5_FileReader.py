import h5py

training_name = input("Enter the training name: ")

reader = h5py.File("./training_datasets/" + training_name + ".hdf5", 'r')
print("Keys:", reader.keys())
print("Features length:", len(reader))

print("Dataset Shape:")
for key in reader.keys():
    print(reader.get(key).shape)

print("First 20 timestamps of the dataset.")
for i in range(0,20):
    print(reader.get("time")[i]/1000000000)

reader.close()