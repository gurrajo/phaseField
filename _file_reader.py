
def file_reader(filename):
    f = open(filename, 'r')
    Lines = f.readlines()
    line = []
    for Line in Lines:
        nums = Line.split("#")
        points = []
        for num in nums:
            if num == "\n":
                line.append([points])
                break
            point = num.replace("[", '')
            point = point.replace("]",'')
            point = point.split(",")
            points.append([float(point[0]), float(point[1])])
    f.close()
    return line


