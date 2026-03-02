import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visual(positive_data,negative_data):
    plt.figure(figsize=(10,7))
    labels=["Positive","Negative"]
    size=[len(positive_data),len(negative_data)]
    plt.pie(size,labels=labels)
    plt.title("Pie chart representation of sentiment analysis")
    plt.savefig("sentiment_pie.png")
