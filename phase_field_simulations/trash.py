import fiona
ident = "A_1-5_k2_1-2MP-s"
schema = {
    'geometry':'LineString',
    'properties':[('Name','str')]
}
polyShp = fiona.open(ident+'Line.shp', mode='w', driver='ESRI Shapefile', schema=schema, crs = "EPSG:4326")
for i, cnt in enumerate(a):
    xylist = cnt.vertices.tolist()
    rowName = i
    rowDict = {
        'geometry': {'type': 'LineString', 'coordinates': xylist},  # Here the xyList is in brackets
        'properties': {'Name': "Line"+str(rowName)},
    }
    print(i)
    polyShp.write(rowDict)

# close fiona object
polyShp.close()