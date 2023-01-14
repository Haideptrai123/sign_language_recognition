import sys
import pickle

x = pickle.load(open('C:/Users/admin/Downloads/Compressed/sign-language-gesture-recognition-master/predicted-frames-final_result-test.pkl', 'rb'))
print("Length", len(x))
print(x)
print("Press y to print all data and n to exit")

t = input()
if t == 'y':
    for each in x:
        print(each)
