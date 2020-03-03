from coffeehouse_dltc.main import DLTC

dltc = DLTC()
dltc.load_model('spam_ham_output')

while True:
    input_text = input("Input: ")
    print(dltc.predict_from_text(input_text))