def convert_one(f, data, F_disc):
    """
    Convert one feature (name f) from continuous to discrete.
    Credit: adapted from lab 6, based on original code from Ameet Soni.
    """

    # first combine the feature values (for f) and the labels
    combineXy = []
    for example in data:
        combineXy.append([example.features[f],example.label])
    combineXy.sort(key=lambda elem: elem[0]) # sort by feature

    # first need to merge uniques
    unique = []
    u_label = {}
    for elem in combineXy:
        if elem[0] not in unique:
            unique.append(elem[0])
            u_label[elem[0]] = elem[1]
        else:
            if u_label[elem[0]] != elem[1]:
                u_label[elem[0]] = None
    # print(unique)
    # print(u_label)

    # find switch points (label changes)
    switch_points = []
    for j in range(len(unique)-1):
        # print(u_label[unique[j]])
        if u_label[unique[j]] != u_label[unique[j+1]] or u_label[unique[j]] \
            == None:
            switch_points.append((float(unique[j])+float(unique[j+1]))/2) # midpoint

    # add a feature for each switch point (keep feature vals as strings)
    for s in switch_points:
        key = f+"<="+str(s)
        for i in range(len(data)):
            if float(data[i].features[f]) <= s:
                data[i].features[key] = 1
            else:
                data[i].features[key] = 0
        # print(key)
        F_disc[key] = [0, 1]
        # print(F_disc[key])

    # delete this feature from all the examples
    for example in data:
        del example.features[f]
