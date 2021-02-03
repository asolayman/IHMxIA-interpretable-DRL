import csv


def embedding2csv(embeddings, actions=None):
    with open("projection.csv", "w") as f:
        spamwriter = csv.writer(f, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if actions is None:
            spamwriter.writerow(("x", "y"))
        else:
            spamwriter.writerow(("x", "y", "arg"))

        i = 0
        for x, y in embeddings:
            if actions is None:
                spamwriter.writerow((x, y))
            else:
                spamwriter.writerow((x, y, actions[i]))
            i += 1
