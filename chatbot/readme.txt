Voice Enabled Chatbot with Machine Learning capabilities

- Data Collection: Conversational data is stored in JSON format. The dataset consists of three parts: tag, pattern, and response. The tag represents the category or topic of the conversation, the pattern captures the input pattern within that category, and the response is what the chatbot should reply with when presented with that intent.

- Data Preprocessing: This involves Tokenization, Stemming, and Vectorization (converting each sentence into bag of words vector) of each pattern sentence. The tags are then one hot encoded, where the position of 1 in a list, indicates respective tag. The sentence vectors and their tags are given as X_train and y_train respectively.

- Neural Network: The neural network model is built using Tensorflow. The architecture of the model consists of an input layer, two hidden layers with 16 neurons each, and an output layer with 7 neurons (for 7 tags). The input layer takes the bag of words as input, and the output layer gives probability of each tag.

- Inference: Once the model is trained, it is used to get probabilities for new input intents. Then a function takes these probabilities, selects the tag with highest probability, and randomly selects a response associated with that tag from given dataset. This response is returned as the reply.

- Speech: The voice part of the bot works using two function. First one, uses speech_recognition library to take voice input through user's microphone. Then it returns transcript using any of the predefined models. This transcript is sent to the main model. 
Second one, uses pyttsx3 library, which takes in the model's reply, and outputs speech using microsoft's inbuilt text-to-speech engine.