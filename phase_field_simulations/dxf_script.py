import ezdxf
from nastran_scripts import _file_reader as fr

lines = fr.file_reader("myfile_4.txt")
# hatch requires the DXF R2000 or later
doc = ezdxf.new("R2000")
msp = doc.modelspace()

# important: major axis >= minor axis (ratio <= 1.)
# minor axis length = major axis length * ratio
#msp.add_ellipse((0, 0), major_axis=(0, 10), ratio=0.5)

# by default a solid fill hatch with fill color=7 (white/black)
#hatch = msp.add_hatch(color=0)
#
# # every boundary path is a 2D element
# # each edge path can contain line, arc, ellipse and spline elements
# # important: major axis >= minor axis (ratio <= 1.)
# hatch.paths.add_lwpolyline([(0,0), (0,255), (255,255), (255,0)]
#     flags=ezdxf.const.BOUNDARY_PATH_EXTERNAL,)
# hatch2 = msp.add_hatch(color=1)
# edge_path = hatch2.paths.add_edge_path()
# edge_path.add_line((0,90), (90,0))
# #edge_path.add_line((0,0), (0,90))
# edge_path.add_line((0,0), (90,5))
msp.add_lwpolyline([(0,0), (0,255), (255,255), (255,0), (0,0)],)
for i, line in enumerate(lines):
    points = line[0]
    msp.add_lwpolyline(points)
doc.saveas("Morph_5.dxf")

#doc.saveas("test.dxf")