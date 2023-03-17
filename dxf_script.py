import ezdxf

doc = ezdxf.readfile("micro_struc_copy.dxf")

doc.saveas("micro_struc_copy.dxf")