import re


def extract_node(line):
    try:
        node = re.search('"(.+?)"', line).group(1)
        node = node.split()
        numbers = [float(n) for n in node]
        return tuple(numbers[:2])

    except AttributeError:
        return None


with open('C:\\Users\\Tim\\Documents\\BeamNG.research\\levels\\asfault\\scenarios\\asfault.prefab', 'r') as file:
    nodes = list()
    for line in file:
        if "new DecalRoad(street_1) {" in line:
            mode = "road"
        elif "new DecalRoad(divider_1_1) {" in line:
            break
        elif "Node = " in line and mode == "road":
            nodes.append(extract_node(line))
    with open('C:\\Users\\Tim\\Documents\\road.xml', 'a') as output:
        for node in nodes:
            output.write('<laneSegment x="{}" y="{}" width="5"/>\n'.format(node[0], node[1]))