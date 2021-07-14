dataset = pandas.read_csv("/home/roboy/workspace/roboy_control/data0.log", delim_whitespace=True, header=1)

dataset = dataset.values[:len(dataset)-1,0:]
np.random.shuffle(dataset)
euler_set = np.array(dataset[:,13:16])