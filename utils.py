

def class_accuracy(target, predicted):
    correct = 0
    for t, p in zip(target, predicted):
        if t.argmax() == p.argmax():
            correct += 1 
    return correct / len(target)

