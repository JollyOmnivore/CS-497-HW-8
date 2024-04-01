import string
import numpy as np
import tensorflow as tf

#Function yoinked from tf website, modified slightly
#this could be done more simply by hand pry easy, check sites
#I sent to you as resources on discord, they should help.
#https://www.tensorflow.org/text/tutorials/transformer
def positional_encoding(length, depth, n=10000):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (n**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

#START HERE
#get file and set vars
encodingFileName = "pangram101.txt"
#geeks4geeks stolen punc value
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
encodingFile = open(encodingFileName, "r", encoding="utf-8")
encodingLine = ""
maxLength = 20

sentences = []
dictionary = []
indDictionary = {}
index = 0

#iterate over pangram101.txt
for encodingLine in encodingFile:
    #data preprocessing:
    #
    #first, we need to remove the punctuation, as pangram101 has punc.
    #this is hevily referencing g4g's code, and I stole their punc value
    #https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
    cleanString = ""
    
    for char in encodingLine:
        if char not in punc:
            cleanString+=char
            
    #then we split into arr of words
    stringArr = cleanString.split()
    
    #append to sentence array, add each unique word to dictionary, and check if new max length
    #also, add indexes to dictionary for later index searching for the OHE (onehot)
    sentences.append(stringArr)
    for word in stringArr:
        if word not in dictionary:
            dictionary.append(word)
        if word not in indDictionary:
            indDictionary.update({word : index + 1})
            index = index + 1
        
#now that we have the scentences, maxLength, and dict, its time for the actual encoding.
#from here on, I'm just trying my best to follow andy's presentation without code
            
#Onehot Encoding is partially stolen from:
#https://medium.com/analytics-vidhya/one-hot-encoding-of-text-data-in-natural-language-processing-2242fefb2148#:~:text=In%20one%20hot%20encoding%2C%20every,one%20hot%20vector%20being%20unique.
#Create np array of size (numSentences, maxLength, totalUniqueWords)
oneHotSentences = np.zeros(shape = (len(sentences),
                                    maxLength,
                                    max(indDictionary.values()) + 1))

#Enumerate lets you iterate over objects with an acompanied iteration val. 
for x, sentence in enumerate(sentences):
    encodedSent = []
    for y, string in enumerate(sentence):
        #find unique index of word in dict and encode to np array
        index = indDictionary.get(string)
        oneHotSentences[x, y, index] = 1 

#Create layer to condense onehot values into a dec float
active = tf.keras.layers.Dense(1, activation='relu')

#run data through layer to reduce the array z dim to 1, and strip dimension
#this leaves us with an array 101x20

def not_my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier
# Hey for this function I had to loked up an alternative to the tf.math.round() function that have a precision value here is were I got it 
# https://stackoverflow.com/questions/46688610/tf-round-to-a-specified-precision
wordEncArr = active(oneHotSentences)
#wordEncArr = round(wordEncArr, 4)
wordEncArr = not_my_tf_round(wordEncArr, 4)


wordEncArr = tf.squeeze(wordEncArr, axis=-1)
print("wordEncArr HERE")
print(wordEncArr)

#Now that we have the word encoded values, we need possition encoding
#using function above, stolen from tf website
possEncArr = positional_encoding(len(wordEncArr), len(wordEncArr[0]))
print(possEncArr)

#Now we add it all up!
encodedArr = []
for i in range(0,len(wordEncArr)):
    encodedMiniArr = []
    for j in range(0,len(wordEncArr[0])):
        encodedVal = wordEncArr[i][j] + possEncArr[i][j]
        encodedMiniArr.append(float(encodedVal.numpy())) #stays tf objects without the .numpy(), but
    encodedArr.append(encodedMiniArr)                    #looses the float values if not casted? IDK,
                                                         #I think tf just likes watching me suffer

#If all went to plan, this should be 101 arrays of 20 float values.
f = open("output_embedding.txt", "w")


for arr in encodedArr:
    for val in arr:
        val = round(val,3)
        print(val)
        f.write(str(val))
        f.write(" ")
f.close()
#Still needs to be done:
#	+Floats need to be between 0-1 (tbd)
#	+Floats must be 4 digits of precision for the word encoding (onehot)
#	+must output to "output_embedding.txt"
#	+output floats must be 3 digits of precision

#Good luck! you got this, this really shouldn't be too bad

