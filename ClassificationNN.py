import numpy as np 
import tensorflow as tf 
import data_reader 



MNIST			= "MNIST"
#https://archive.ics.uci.edu/ml/datasets/Poker+Hand
#Different 5-card poker-hands are input into the NN and classified as one of 10 different poker-hands. 97% accuracy 
POKER_HANDS 	= "POKER_HANDS"
#https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
BANK			= "BANK"	
#https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
#89% accuracy
HAR 			= "HAR"
#https://archive.ics.uci.edu/ml/datasets/Wine+Quality, 
#note that only 50% accuracy has been acheived with this dataset, perhaps it can be improved. 
WINE			= "WINE" 

class ClassificationNN:


	def __init__(self,dataset,
		learning_rate=0.015,
		training_iteration=30,
		batch_size=50,
		hidden_layer_n=255):

		if (dataset == MNIST):
			self.dataset = data_reader.MNIST()
		if (dataset == POKER_HANDS):
			self.dataset = data_reader.poker_hand()
		if (dataset == HAR):
			self.dataset = data_reader.har()
		if (dataset == WINE):
			self.dataset = data_reader.wine_quality()
		if (dataset == BANK):
			self.dataset = data_reader.bank()

		self.learning_rate 		= learning_rate
		self.training_iteration = training_iteration
		self.batch_size 		= batch_size
		input_layer_n 			= self.dataset.input_dim
		output_layer_n 			= self.dataset.output_dim
		self.display_step		= 3

		self.x = tf.placeholder("float", [None, input_layer_n], name="x")
		self.y = tf.placeholder("float", [None, output_layer_n], name="y")

		with tf.name_scope("weights") as scope:
			W1 = tf.Variable(tf.random_normal([input_layer_n, hidden_layer_n], stddev=0.1))
			W2 = tf.Variable(tf.random_normal([hidden_layer_n, output_layer_n], stddev=0.1))



		with tf.name_scope("biases") as scope:
			b1 = tf.Variable(tf.random_normal([hidden_layer_n], stddev=0.1))
			b2 = tf.Variable(tf.random_normal([output_layer_n], stddev=0.1))

		#Histograms for weights and biases
		weights_histogram1 	= tf.summary.histogram("weights 1", W1) 
		biases_histogram1 	= tf.summary.histogram("biases 1", b1)
		weights_histogram2 	= tf.summary.histogram("weights 2", W2) 
		biases_histogram2	= tf.summary.histogram("biases 2", b2)

		with tf.name_scope("model") as scope:
			layer_1 = tf.nn.sigmoid(tf.matmul(self.x, W1) + b1)
			layer_2 = tf.nn.softmax(tf.matmul(layer_1, W2) + b2)
			self.model = layer_2


		with tf.name_scope("objective_function") as scope:
			self.objective_function = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.model, self.y))))
			tf.summary.scalar("objective_function", self.objective_function)

		with tf.name_scope("train") as scope:
			self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.objective_function)


		self.init = tf.global_variables_initializer()
		self.merged_summary_op = tf.summary.merge_all()


	def run(self):
		with tf.Session() as sess:
			sess.run(self.init)
			summary_writer = tf.summary.FileWriter('data/logs', graph=sess.graph)
			for iteration in range(self.training_iteration):
				avg_cost = 0
				num_batches = int(self.dataset.num_examples/self.batch_size)
				for i in range(num_batches):
					batch_xs, batch_ys = self.dataset.next_batch(self.batch_size)
					sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys})
					avg_cost += sess.run(self.objective_function, feed_dict={self.x: batch_xs, self.y: batch_ys})/num_batches
					summary_str = sess.run(self.merged_summary_op, feed_dict={self.x: batch_xs, self.y: batch_ys})
					summary_writer.add_summary(summary_str, iteration*num_batches + i)
				if iteration % self.display_step == 0:
					print("Epoch:" + str(iteration+1) +  " cost: ", "{:.9f}".format(avg_cost))

			print("Training completed!")
  
			predictions = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
			accuracy = tf.reduce_mean(tf.cast(predictions,  tf.float64))
			print("Testing accuracy:", accuracy.eval({self.x: self.dataset.testing_X, self.y: self.dataset.testing_y}))

if __name__== "__main__":

	#nn = ClassificationNN(MNIST)
	nn = ClassificationNN(POKER_HANDS, learning_rate=0.015,training_iteration=250)
	#nn = ClassificationNN(HAR, learning_rate=0.015,batch_size=50,training_iteration=100, hidden_layer_n=200)
	#nn = ClassificationNN(WINE, learning_rate=0.015,batch_size=100,training_iteration=1000)
	#nn = ClassificationNN(BANK, learning_rate=0.01,batch_size=100,training_iteration=50)
	nn.run()
	#Regularization - balance improvement of accuracy with the balanch of computation 


