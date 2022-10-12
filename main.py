import pickle 
import numpy

def main(): 
    with open('datafiles/assignment-one-test-parameters.pkl', 'rb') as f: 
        data = pickle.load(f)
        print(data)

if __name__ == '__main__': 
    main()