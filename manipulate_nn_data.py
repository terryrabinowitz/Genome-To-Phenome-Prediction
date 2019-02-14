import numpy as np
import csv
np.set_printoptions(suppress=True)

aa_hash = {'a': 0, 'c': 1, 'd': 2, 'e': 3, 'f': 4, 'g': 5, 'h': 6, 'i': 7, 'k': 8, 'l': 9, 'm': 10, 'n': 11, 'p': 12,
           'q': 13, 'r': 14, 's': 15, 't': 16, 'v': 17, 'w': 18, 'y': 19}


def homogenous_input(path):
      load_path = path + "Henryetal_SuppMat2.csv"
      save_path = path + 'Henryetal_SuppMat2.npy'
      f = open(load_path, "r")
      header = f.readline()
      header = header.split()
      contents = f.readlines()
      num_samples = len(contents)
      num_snps = len(header) - 1  # remove one since the first value of the header is just the column label
      num_amino_acids = 20
      data = np.zeros((num_samples, num_snps, num_amino_acids))
      strains = []
      for strain_counter in range(len(contents)):
            line = contents[strain_counter]
            line = line.strip()
            words = line.split()
            strain = words[0]
            snps = words[1:len(words)]
            strains.append(strain)
            for snp_counter in range(len(snps)):
                  aa = snps[snp_counter].lower()
                  aa_counter = aa_hash[aa]
                  data[strain_counter, snp_counter, aa_counter] = 1.0
      np.save(save_path, data)


def heterogeneous_input(path):
      # NOTE - this is a contrived file just to get the function working
      load_path = path + 'Henryetal_SuppMat2_ds.csv'
      save_path = path + 'Henryetal_SuppMat2_ds.npy'
      f = open(load_path, "r")
      header = f.readline()
      header = header.split()
      contents = f.readlines()
      num_samples = len(contents) / 2
      num_strands = 2
      num_snps = len(header) - 1  # remove one since the first value of the header is just the column label
      num_amino_acids = 20
      data = np.zeros((num_samples, num_strands, num_snps, num_amino_acids))
      strains = []

      for strain_counter in range(len(contents)):
            line = contents[strain_counter]
            line = line.strip()
            words = line.split()
            strain = words[0]
            snps = words[1:len(words)]
            if strain_counter % 2 == 0:
                  strand = 0
                  strains.append(strain)
            else:
                  strand = 1
            strain_counter_real = (strain_counter / 2)
            for snp_counter in range(len(snps)):
                  aa = snps[snp_counter].lower()
                  aa_counter = aa_hash[aa]
                  data[strain_counter_real, strand, snp_counter, aa_counter] = 1.0
            np.save(save_path, data)



def real_input(path):
      # NOTE - this takes the homo data and duplicates it for each strand
      load_path = path + 'Henryetal_SuppMat1.csv'
      f = open(load_path, "r")
      header = f.readline()
      header = header.split()
      contents = f.readlines()
      num_samples = len(contents)
      num_strands = 2
      num_snps = len(header) - 1  # remove one since the first value of the header is just the column label
      num_amino_acids = 20
      strains = []
      data = np.zeros((num_samples, num_strands, num_snps, num_amino_acids))
      for strain_counter in range(len(contents)):
            line = contents[strain_counter]
            line = line.strip()
            words = line.split()
            strain = words[0]
            strains.append(strain)
            snps = words[1:len(words)]
            for snp_counter in range(len(snps)):
                  aa = snps[snp_counter].lower()
                  aa_counter = aa_hash[aa]
                  data[strain_counter, 0:2, snp_counter, aa_counter] = 1.0
      return data, strains

def generate_model_sets(data_in, data_out, strains, path):
      save_path_all_in = path + 'ALL_INPUT.npy'
      save_path_all_out = path + 'ALL_OUTPUT.npy'
      save_path_all_strain = path + 'ALL_STRAINS.txt'


      save_path_train_in = path + 'TRAIN_INPUT.npy'
      save_path_val_in= path + 'VALIDATE_INPUT.npy'
      save_path_test_in = path + 'TEST_INPUT.npy'
      save_path_train_out= path + 'TRAIN_OUTPUT.npy'
      save_path_val_out= path + 'VALIDATE_OUTPUT.npy'
      save_path_test_out = path + 'TEST_OUTPUT.npy'
      save_path_train_strain= path + 'TRAIN_STRAINS.txt'
      save_path_val_strain= path + 'VALIDATE_STRAINS.txt'
      save_path_test_strain = path + 'TEST_STRAINS.txt'
      #shuffle_in_unison_2(data_in, data_out)
      train_data_in = data_in[:42]
      val_data_in = data_in[42:63]
      test_data_in = data_in[63:]
      train_data_out = data_out[:42]
      val_data_out = data_out[42:63]
      test_data_out = data_out[63:]
      train_data_strains = strains[:42]
      val_data_strains = strains[42:63]
      test_data_strains = strains[63:]
      np.save(save_path_train_in, train_data_in)
      np.save(save_path_val_in, val_data_in)
      np.save(save_path_test_in, test_data_in)
      np.save(save_path_train_out, train_data_out)
      np.save(save_path_val_out, val_data_out)
      np.save(save_path_test_out, test_data_out)
      np.save(save_path_all_in, data_in)
      np.save(save_path_all_out, data_out)

      f = open(save_path_train_strain, "w")
      for i in train_data_strains:
            i = i + "\n"
            f.write(i)
      f = open(save_path_val_strain, "w")
      for i in val_data_strains:
            i = i + "\n"
            f.write(i)
      f = open(save_path_test_strain, "w")
      for i in test_data_strains:
            i = i + "\n"
            f.write(i)
      f = open(save_path_all_strain, "w")
      for i in strains:
            i = i + "\n"
            f.write(i)



def real_output(path):
      load_path = path + 'SupplementaryTable1.csv'
      f = open(load_path, "r")
      f= csv.reader(f)
      temp = []
      for row in f:
            temp.append(row)
      num_samples = len(temp) -1
      num_traits = len(temp[0]) - 3  # remove three metadata points in the header
      data = np.zeros((num_samples, num_traits))

      for strain_counter in range(1, len(temp)):
            for trait_counter in range(3,len(temp[strain_counter])):
                  data[strain_counter-1, trait_counter-3] = temp[strain_counter][trait_counter]
      return data

def simulated_offspring_input(path):
      load_path = path + "TRAIN_INPUT.npy"
      save_path = path + "SIMULATED_OFFSPRING_TEST_INPUT"
      parents = np.load(load_path)
      offspring = []
      for strain_counter_i in range(len(parents)-1):
            for strain_counter_j in range(strain_counter_i+1,len(parents)):
                  parent_i = parents[strain_counter_i]
                  parent_j = parents[strain_counter_j]
                  homogenous_i = np.array_equal(parent_i[0], parent_i[1])
                  homogenous_j = np.array_equal(parent_j[0], parent_j[1])
                  if (homogenous_i and homogenous_j):
                        child = np.stack([parent_i[0],parent_j[0]])
                        offspring.append(child)
                  elif (homogenous_i and not homogenous_j):
                        child1 = np.stack([parent_i[0],parent_j[0]])
                        child2 = np.stack([parent_i[0],parent_j[1]])
                        offspring.append(child1)
                        offspring.append(child2)
                  elif (not homogenous_i and homogenous_j):
                        child1 = np.stack([parent_i[0],parent_j[0]])
                        child2 = np.stack([parent_i[1],parent_j[0]])
                        offspring.append(child1)
                        offspring.append(child2)
                  else:
                        child1 = np.stack([parent_i[0],parent_j[0]])
                        child2 = np.stack([parent_i[1],parent_j[0]])
                        child3 = np.stack([parent_i[0],parent_j[1]])
                        child4 = np.stack([parent_i[1],parent_j[1]])
                        offspring.append(child1)
                        offspring.append(child2)
                        offspring.append(child3)
                        offspring.append(child4)
      offspring = np.asarray(offspring)

      np.save(save_path, offspring)

def example_output(path):
      save_path = path + 'VALIDATE_OUTPUT.npy'
      num_samples = 5
      num_traits = 9
      data = np.random.rand(num_samples, num_traits) * 5
      total = np.expand_dims(np.sum(data, axis =1),axis=1)
      combine = np.concatenate((data,total), axis=1)
      np.save(save_path, combine)


def example_output_relative(path):
      load_path = path + 'TEST_OUTPUT.npy'
      save_path = path + 'TEST_OUTPUT_2.npy'
      data_abs = np.load(load_path)
      num_samples =  data_abs.shape[0]
      num_traits = data_abs.shape[1] - 1
      data_rel = np.zeros((num_samples, num_traits))
      for strain_counter in range(num_samples):
            for trait_counter in range(num_traits):
                  data_rel[strain_counter][trait_counter] = data_abs[strain_counter][trait_counter] / data_abs[strain_counter][num_traits]
      np.save(save_path, data_rel)
      print data_rel

def shuffle_in_unison_2(a, b):
      rng_state = np.random.get_state()
      np.random.shuffle(a)
      np.random.set_state(rng_state)
      np.random.shuffle(b)


#####################################################

path = '/Users/terryrabinowitz/PycharmProjects/cannabis/data/real/'
# homogenous_input(path)
# heterogeneous_input(path)
# example_output(path)
#example_output_relative(path)
# simulated_offspring_input(path)

data_input, strains = real_input(path)
print  data_input.shape
data_output = real_output(path)
print data_output.shape
generate_model_sets(data_input, data_output,strains, path)