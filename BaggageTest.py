import yaml
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import xml.etree.ElementTree as ET
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import read_image
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv

def load_Config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


file_config = load_Config("configs/paths.yaml")

def get_image_dataframe_lazy(image_folder, name):
    """
    Create a DataFrame with image metadata only (NO tensors).
    Images are loaded lazily when needed.
    """

    image_stats = []

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(".jpg"):
            image_path = os.path.join(image_folder, image_file)

            try:
                # Use PIL to get metadata WITHOUT loading full tensor
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode

                image_stats.append([
                    image_file,
                    image_path,
                    width,
                    height,
                    mode
                ])

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    df = pd.DataFrame(
        image_stats,
        columns=["filename", "path", "width", "height", "mode"]
    )

    print(f"--- Stats for {name} ---")
    print(f"Unique Images: {df['filename'].nunique()}")
    print("\nSummary of Image Dimensions:")
    print(df[["width", "height"]].describe())
    print("\n" * 3)

    return df

def compute_csv_stats(csv_path, name):
    """
    Load a CSV file, compute basic statistics, and return the data as a DataFrame.

    Parameters:
    csv_path (str): The path to the CSV file containing annotations.
    name (str): Name to use in the printed output for the dataset.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded CSV data with computed stats.
    Returns None if an error occurs while loading the CSV.
    """

    try:
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_path)

        # Display basic stats
        print(f"--- Stats for {name} ---")
        print(f"Total Rows: {len(df)}")
        print(f"Unique Images: {df['filename'].nunique()}")

        # Show a summary of the dataset (describe numeric fields)
        print("\nSummary Statistics:")
        print(df.describe())

        # Add a blank line after each section for readability
        print("\n" * 3)

        # Return the DataFrame for later use
        return df

    except Exception as e:
        print(f"An error occurred while processing {name}: {e}")
        return None  # Return None if there's an error

def create_csv_if_not_exists(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename","width","height","depth",
                "class",
                "xmin","ymin","xmax","ymax"
            ])

class XRayImageDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = read_image(row["path"])
        return image, row["filename"]

df_train_images = get_image_dataframe_lazy(
    file_config["paths"]["trainingImages"], "Training Images"
)
df_test_images = get_image_dataframe_lazy(
    file_config["paths"]["testImages"], "Testing Images"
)

df_train_annotations = compute_csv_stats(file_config["paths"]["trainningAnnot_file"], 'Training Annotations')
df_test_annotations = compute_csv_stats(file_config["paths"]["testAnnot_file"], 'Testing Annotations')

df_train_annotations.head()

#Remove white

torch_to_np = lambda tse : tse.permute(1, 2, 0).numpy()
np_to_torch = lambda npa : torch.from_numpy(npa).permute(2, 0, 1)

def show_image(pic, convert = False):
  print(f"dtype : {pic.dtype}")
  print(f"Shape : {pic.shape}")
  print(f"Aspect Ratio (h/w) : {round(pic.shape[0]/pic.shape[1], 2)}")
  plt.imshow(pic)
  plt.show()
  

def get_whiteness(im, whiteness_tolerance = 30, mean = (104, 117, 123)):
  mn_np = np.array(mean)
  if isinstance(im, torch.Tensor):
    im = im.numpy()
  im  = im + mn_np
  im = im.reshape((im.shape[0]*im.shape[1],3))
  white_count = 0
  for tup in im:
    if (255-sum(tup)/3 <= whiteness_tolerance):
    # if (sum(tup)/3 <= whiteness_tolerance):
      white_count += 1
  white_per = 100*white_count/len(im)
  # print(f"white_per = {white_per}")
  return round(white_per, 2)

class BaseTransform:
    def __init__(self, resize= (584, 688), rgb_means=(104, 117, 123), transform_ = "medium"):
        self.resize = resize
        self.rgb_means = torch.tensor(rgb_means).view(3, 1, 1)
        self.transform_ = transform_

    def __call__(self, img):
        img = self.resize_image(img)

        if self.transform_ == "light":
            img = img + self.rgb_means
        elif self.transform_ == "dark":
            img = img - self.rgb_means

        return img

    def resize_image(self, img):
        img = F.interpolate(img.unsqueeze(0), size= self.resize, mode='bilinear', align_corners=False).squeeze(0)
        return img

def reflect(old_w, w_max):
    mid_w = w_max//2
    if(mid_w < old_w):
        new_w = old_w - 2*(old_w - mid_w)
    else:
        new_w = old_w + 2*(mid_w - old_w)
    return new_w

def get_nearest_obj(tran_dic, seg_X):
  seg_X = int(seg_X)
  if (tran_dic[seg_X] != -1):
    return seg_X
  bfl = list(tran_dic.keys())[:seg_X]
  bfl = bfl[::-1]
  bf_counter = 0
  bf_seg = -1
  for bf in bfl:
    bf_counter += 1
    if (tran_dic[bf] != -1):
      bf_seg = bf
      break
  if (bf_seg == -1):
    bf_counter = 9999

  afl = list(tran_dic.keys())[seg_X+1:]
  af_counter = 0
  af_seg = -1
  for af in afl:
    af_counter += 1
    if (tran_dic[af] != -1):
      af_seg = af
      break
  if (af_seg == -1):
    af_counter = 9999

  if (bf_counter <= af_counter):
    new_seg = bf_seg
  else:
    new_seg = af_seg
  return new_seg

def rotated_this(old_x, old_y, xmax, ymax):
  new_org_x, new_org_y = (0, xmax)
  return (new_org_x+old_y, new_org_y-old_x)

def remove_white(im, do_vertical = True, do_horizontal = True, v_whiteness_threshold = 80, h_whiteness_threshold = 80, seg_wid = 10, whiteness_tolerance = 33):
  if (do_vertical):
    img_hig = im.shape[0]
    num_of_seg = int(img_hig/seg_wid)
    seg_num, iter, br, all_where = 0, 0, 0, 0
    v_dic = dict(zip([i for i in range(num_of_seg)], [-1 for i in range(num_of_seg)]))
    while (seg_num <= num_of_seg-br+2):
      if (all_where >= img_hig):
        break
      this_seg = im[seg_num*seg_wid:seg_num*seg_wid+seg_wid,:]
      # print(f"VER this_seg = {this_seg.shape}") ###############################
      if (get_whiteness(this_seg, whiteness_tolerance)>v_whiteness_threshold):
        br += 1
        im = np.append(im[:seg_num*seg_wid,:], im[seg_num*seg_wid+seg_wid:,:], 0)
      else:
        v_dic[iter] = iter - br
        seg_num += 1
      iter += 1
      all_where += seg_wid

  # print(f"BEFORE HOR shape of im.shape = {im.shape}")
  if (do_horizontal):
    img_wid = im.shape[1]
    num_of_seg = int(img_wid/seg_wid)
    seg_num, iter, br, all_where = 0, 0, 0, 0
    h_dic = dict(zip([i for i in range(num_of_seg)], [-1 for i in range(num_of_seg)]))
    while (seg_num <= num_of_seg-br+2):
      if (all_where >= img_wid):
        break
      this_seg = im[:,seg_num*seg_wid:seg_num*seg_wid+seg_wid]
      # print(f"HOR this_seg = {this_seg.shape}") ###############################
      if (get_whiteness(this_seg, whiteness_tolerance)>h_whiteness_threshold):
        br += 1
        im = np.append(im[:,:seg_num*seg_wid], im[:,seg_num*seg_wid+seg_wid:], 1)
      else:
        h_dic[iter] = iter - br
        seg_num += 1
      iter += 1
      all_where += seg_wid
  max_val_v_dic = max(v_dic.values())
  for k, v in v_dic.items():
    if (v == 0):
      for i in range(k):
        v_dic[i] = 0
    if (v == max_val_v_dic):
      for i in range(k, len(v_dic)):
        v_dic[i] = max_val_v_dic
  max_val_h_dic = max(h_dic.values())
  for k, v in h_dic.items():
    if (v == 0):
      for i in range(k):
        h_dic[i] = 0
    if (v == max_val_h_dic):
      for i in range(k, len(h_dic)):
        h_dic[i] = max_val_h_dic
    return (seg_wid, h_dic, v_dic, im)

def save_tensor_as_image(tensor, path):
  if tensor.ndim == 3:   # C,H,W
      tensor = tensor.permute(1, 2, 0)

  img = tensor.detach().cpu().numpy()

  if img.max() <= 1.0:
      img = (img * 255).astype(np.uint8)
  else:
      img = img.astype(np.uint8)

  cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
  
  return img.shape,"grayscale" if img.ndim == 2 else "RGB"

## RESIZING TEST IMAGES

test_count = len(df_test_images)
image_tensor_list = []
filenames_list = []
object_locations_list = []

new_image_crop = []
csv_path = "Temp\Test\_annotationstest.csv"
create_csv_if_not_exists(csv_path)
for fi in range(test_count):
  ######################################################################################################
  ################################################### ORIGINAL ###################################################
  ######################################################################################################  
  ann_pic_path = df_test_images['path'][fi]
  ann_pic_name = df_test_images['filename'][fi]
  
  
  ann_pic = read_image(ann_pic_path)
  # print(f"{fi}/{test_count} <> {ann_pic_name}")

  # show_image(torch_to_np(ann_pic))
  filenames_list.append(ann_pic_name)



  ######################################################################################################
  ################################################### ORIGINAL WITH BOXES ###################################################
  ######################################################################################################


  # xml_file_path = os.path.join(r"C:\Users\Gaura\OneDrive\Desktop\GAURAV\Unimleb\COMP90051\Statisitcal_Machine_Learning_Project\XRay\Train\Annotations", ann_pic_name[:-4] + ".xml")
  xml_file_path = os.path.join('Test', 'Annotations', ann_pic_name[:-4] + ".xml")

  tree = ET.parse(xml_file_path)
  root = tree.getroot()

  # i = 0
  # obj_id = []
  # for child in root:
  #   if (child.tag == "object"):
  #     obj_id.append(i)
  #   i += 1

  # dim_dic = {}
  # count = 0
  # for i in obj_id:
  #   dim_dic[count] = {}
  #   dim_dic[count]["xmin"] = int(root[i][4][0].text)
  #   dim_dic[count]["ymin"] = int(root[i][4][1].text)
  #   dim_dic[count]["xmax"] = int(root[i][4][2].text)
  #   dim_dic[count]["ymax"] = int(root[i][4][3].text)
  #   dim_dic[count]["name"] = root[i][0].text
  #   count += 1
  # dim_dic

  # boxes = []
  # labels = []
  # for key, value in dim_dic.items():
  #     boxes.append([value['xmin'], value['ymin'], value['xmax'], value['ymax']])
  #     labels.append(value['name'])

  # boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
  # image_with_boxes = draw_bounding_boxes(ann_pic, boxes_tensor, labels=labels, colors="red", width=3)

  # show_image(torch_to_np(image_with_boxes))

  # image_with_boxes_pil = torchvision.transforms.ToPILImage()(image_with_boxes)
  # plt.imshow(image_with_boxes_pil)
  # plt.title("Image with Bounding Boxes")
  # plt.axis("off")  
  # plt.show()


  ######################################################################################################
  ################################################### NO WHITE SPACES ###################################################
  ######################################################################################################

  hld = 99.9
  seg_size, h_tran, v_tran, no_whi_np = remove_white(im = torch_to_np(ann_pic), v_whiteness_threshold = hld, h_whiteness_threshold = hld, seg_wid = 1)
  # show_image(no_whi_np)


  ########## EXTRACTING object location from annotation files ##########
  i = 0
  obj_id = []
  for child in root:
    if (child.tag == "object"):
      obj_id.append(i)
    i += 1

  dim_dic = {}
  count = 0
  for i in obj_id:
    dim_dic[count] = {}
    dim_dic[count]["xmin"] = int(root[i][4][0].text)
    dim_dic[count]["ymin"] = int(root[i][4][1].text)
    dim_dic[count]["xmax"] = int(root[i][4][2].text)
    dim_dic[count]["ymax"] = int(root[i][4][3].text)
    dim_dic[count]["name"] = root[i][0].text
    count += 1

  
  bbox0_list = []
  for bb in dim_dic.values():
    temp = []
    temp.append(bb["ymin"])
    temp.append(bb["xmin"])
    temp.append(bb["ymax"])
    temp.append(bb["xmax"])
    bbox0_list.append(temp)
  bbox0 = np.array(bbox0_list)
  c_bbox = bbox0

  ######################################################################################################
  ################################################### Objects location after Removing white space ###################################################
  ##################################################################################################


  w_bbox = []
  for box in c_bbox: 
    temp_box = []
    y_min = box[0]
    x_min = box[1]
    y_max = box[2]
    x_max = box[3]
    y_min_seg = y_min//seg_size
    x_min_seg = x_min//seg_size  
    y_max_seg = y_max//seg_size
    x_max_seg = x_max//seg_size

    # y_min_seg = y_min//seg_size
    # x_min_seg = x_min//seg_size  
    # y_max_seg = y_max//seg_size
    # x_max_seg = x_max//seg_size

    if (x_min_seg <= 0):
      x_min_seg = min(h_tran.keys())
    if (y_min_seg <= 0):
      y_min_seg = min(v_tran.keys())
    if (x_max_seg >= len(h_tran)):
      x_max_seg = max(h_tran.keys())
    if (y_max_seg >= len(v_tran)):
      y_max_seg = max(v_tran.keys())

    if (v_tran[y_min_seg] == -1):
      y_min_seg = get_nearest_obj(tran_dic = v_tran, seg_X = y_min_seg)
    if (h_tran[x_min_seg] == -1):
      x_min_seg = get_nearest_obj(tran_dic = h_tran, seg_X = x_min_seg)
    if (v_tran[y_max_seg] == -1):
      y_max_seg = get_nearest_obj(tran_dic = v_tran, seg_X = y_max_seg)
    if (h_tran[x_max_seg] == -1):
      x_max_seg = get_nearest_obj(tran_dic = h_tran, seg_X = x_max_seg)

    w_y_min = seg_size*v_tran[y_min_seg] + y_min%seg_size
    w_x_min = seg_size*h_tran[x_min_seg] + x_min%seg_size
    w_y_max = seg_size*v_tran[y_max_seg] + y_max%seg_size
    w_x_max = seg_size*h_tran[x_max_seg] + x_max%seg_size

    temp_box.append(w_y_min)
    temp_box.append(w_x_min)
    temp_box.append(w_y_max)
    temp_box.append(w_x_max)
    w_bbox.append(temp_box)

  # boxes = []
  labels = []
  for key, value in dim_dic.items():
      # boxes.append([value['xmin'], value['ymin'], value['xmax'], value['ymax']])
      labels.append(value['name'])

  rot_c_bbox = []
  for box in w_bbox:
      rot_x1, rot_y1 = rotated_this(old_x = box[1], old_y =box[0], xmax=no_whi_np.shape[1], ymax=no_whi_np.shape[0])
      rot_x2, rot_y2 = rotated_this(old_x = box[3], old_y =box[2], xmax=no_whi_np.shape[1], ymax=no_whi_np.shape[0])
      rot_c_bbox.append([ rot_y2, rot_x1, rot_y1, rot_x2])
  W_boxes = [[int(x) for x in row] for row in rot_c_bbox]


  h_max, w_max, _ = no_whi_np.shape
  res = []
  for bx in W_boxes:
      wmin_, hmin_, wmax_, hmax_ = bx[0], bx[1], bx[2], bx[3],
      this_box_X = [reflect(old_w = wmax_, w_max = w_max),  hmin_, reflect(old_w = wmin_, w_max = w_max),  hmax_]
      res.append(this_box_X)

  W_boxes = res

  # boxes_tensor = torch.tensor(W_boxes, dtype=torch.float32)
  # image_with_boxes = draw_bounding_boxes(np_to_torch(no_whi_np), boxes_tensor, labels=labels, colors="red", width=3)
  # show_image(torch_to_np(image_with_boxes))

  ######################################################################################################
  ################################################### Resizing to 584 x 688 ###################################################
  ##################################################################################################

  # transform = BaseTransform(transform_= "light")
  transform = BaseTransform(transform_= "medium")
  # transform = BaseTransform(transform_= "dark")
  transformed_pic = transform(np_to_torch(no_whi_np))

  transformed_pic_np = torch_to_np(transformed_pic)
  Hratio = transformed_pic_np.shape[1]/no_whi_np.shape[1]
  Wratio = transformed_pic_np.shape[0]/no_whi_np.shape[0]

  ratioLst = [Hratio, Wratio, Hratio, Wratio]
  w_bbox_r = []
  for box in W_boxes:
      box = [abs(int(a * b)) for a, b in zip(box, ratioLst)] 
      w_bbox_r.append(box)
  
  final_obj_location_dict = {}
  for label_index in range(len(labels)):
    final_obj_location_dict[label_index] = [labels[label_index], w_bbox_r[label_index]]
    
  os.makedirs("Temp\Test\Images", exist_ok=True)
  
  filePath = "Temp\Test\Images" + ann_pic_name
  shape,mode = save_tensor_as_image(transformed_pic,filePath)
  new_image_crop.append([ann_pic_name,filePath,shape[1],shape[0],mode])
  with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    for k in final_obj_location_dict:
      label, box = final_obj_location_dict[k]
      ymin = box[0]
      xmin = box[1]
      ymax = box[2]
      xmax = box[3]
            
      writer.writerow([
        ann_pic_name,
        shape[1],
        shape[0],
        shape[2],
        label,
        xmin,
        ymin,
        xmax,
        ymax
      ])
    

  #image_tensor_list.append(transformed_pic)
  #object_locations_list.append(final_obj_location_dict)
  print(f"{fi}/{test_count} <> {ann_pic_name} <> final_obj_location_dict = {final_obj_location_dict}")
  # boxes_tensor = torch.tensor(w_bbox_r, dtype=torch.float32)
  # resized_image_with_boxes = draw_bounding_boxes(transformed_pic, boxes_tensor, labels=labels, colors="red", width=3)
  # show_image(torch_to_np(resized_image_with_boxes), convert = False)
  

