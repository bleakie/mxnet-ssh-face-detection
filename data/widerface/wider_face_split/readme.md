Attached the mappings between attribute names and label values.

blur:
  clear->0
  normal blur->1
  heavy blur->2

occlusion:
  no occlusion->0
  partial occlusion->1
  heavy occlusion->2

invalid:
  false->0(valid image)
  true->1(invalid image)

The format of txt ground truth.
File name
Number of bounding box
x1, y1, w, h, blur, invalid, occlusion


为了利用widerface更多的标注信息，我对读取的内容进行了修改，只读取x1, y1, w, h, blur, invalid, occlusion这几个信息，命名为wider_face_train_bbx_gt_clean.txt
其中blur, invalid, occlusion数值越大表示图片的hard程度越高
代码中blur>1, invalid>0, occlusion>1的人脸都加上mask
