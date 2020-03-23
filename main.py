from joblib import load
import pickle 

def label(result):
    #load labels
    pkl_file = open('labels.pkl', 'rb')
    le_departure = pickle.load(pkl_file) 
    pkl_file.close()
    for i in result:
        print("Marca: %s" %(le_departure[i]))

def main(marca):
    #load model
    filename = 'model.sav'
    loaded_model = load(filename)
    result = loaded_model.predict([marca])

    label(result)
  
if __name__== "__main__":
    i = input("Digite o nome do estabelecimento que vem descrito:")
    main(i)





