import os
import sys
import json
import cv2
import numpy as np
from hausdorff_dist import hausdorff_distance
from collections import Counter
from collections import defaultdict
import math 

def load_tracks_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    tracks = {}
    for trackid, track_data in data.items():
        # Convert trackid to int
        trackid = int(trackid)
        
        # Get start and end frames
        startframe = min(track_data['frameid'])
        endframe = max(track_data['frameid'])
        
        # Get most common class (voting mechanism)
        class_counts = Counter(track_data['class'])
        most_common_class = class_counts.most_common(1)[0][0]
        
        # Create bbox list
        bbox = []
        for i, frameid in enumerate(track_data['frameid']):
            x1, y1, x2, y2 = track_data['box'][i]
            bbox.append([frameid, int(x1), int(y1), int(x2), int(y2), most_common_class])
        
        # Create track entry
        tracks[trackid] = {
            'startframe': startframe,
            'endframe': endframe,
            'bbox': bbox,
            'tracklet': track_data['tracklets']
        }
    
    return tracks


def check_bbox_inside_with_roi(bbox, mask):
    #check if four point of bbox all in roi area
    is_inside = True

    x_tl = bbox[1]
    y_tl = bbox[2]
    x_br = bbox[3]
    y_br = bbox[4]

    for x in [x_tl, x_br]:
        if x <= 0 or x >= mask.shape[1]:
            return False

    for y in [y_tl, y_br]:
        if y <= 0 or y >= mask.shape[0]:
            return False

    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:
        (g, b, r) = mask[v[1], v[0]]
        if (g, b, r) < (128, 128, 128):
            is_inside = False
            return is_inside

    return is_inside

def check_tracks_with_roi(tracks, mask):
    tracks_end_in_roi = []
    tracks_start_in_roi = []
    tracks_too_short = []

    for trackid, track in tracks.items():
        start_bbox = track['bbox'][0]
        end_bbox = track['bbox'][-1]

        if check_bbox_inside_with_roi(start_bbox, mask) == True:
            if track['startframe'] > 3:
                tracks_start_in_roi.append(trackid)

        if check_bbox_inside_with_roi(end_bbox, mask) == True:
            tracks_end_in_roi.append(trackid)

        if track['endframe'] - track['startframe'] < 10:
            if trackid not in tracks_start_in_roi:
                tracks_too_short.append(trackid)
    return tracks_end_in_roi, tracks_start_in_roi, tracks_too_short


def check_bbox_overlap_with_roi(bbox, mask):
    is_overlap = False
    if bbox[1] >= mask.shape[1] or bbox[2] >= mask.shape[0] \
            or bbox[3] < 0 or bbox[4] < 0:
        return is_overlap

    x_tl = bbox[1] if bbox[1] > 0 else 0
    y_tl = bbox[2] if bbox[2] > 0 else 0
    x_br = bbox[3] if bbox[3] < mask.shape[1] else mask.shape[1] - 1
    y_br = bbox[4] if bbox[4] < mask.shape[0] else mask.shape[0] - 1
    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:
        (g, b, r) = mask[v[1], v[0]]
        if (g, b, r) > (128, 128, 128):
            is_overlap = True
            return is_overlap

    return is_overlap

def is_same_direction(traj1, traj2, angle_thr):
    vec1 = np.array([traj1[-1][0] - traj1[0][0], traj1[-1][1] - traj1[0][1]])
    vec2 = np.array([traj2[-1][0] - traj2[0][0], traj2[-1][1] - traj2[0][1]])
    L1 = np.sqrt(vec1.dot(vec1))
    L2 = np.sqrt(vec2.dot(vec2))
    if L1 == 0 or L2 == 0:
        return False
    cos = vec1.dot(vec2)/(L1*L2)
    angle = np.arccos(cos) * 360/(2*np.pi)
    if angle < angle_thr:
        return True
    else:
        return False

def calc_angle(traj1, traj2):
    vec1 = np.array([traj1[-1][0] - traj1[-5][0], traj1[-1][1] - traj1[-5][1]])
    vec2 = np.array([traj2[-1][0] - traj2[-5][0], traj2[-1][1] - traj2[-5][1]])
    L1 = np.sqrt(vec1.dot(vec1))
    L2 = np.sqrt(vec2.dot(vec2))
    if L1 == 0 or L2 == 0:
        return 90
    cos = vec1.dot(vec2)/(L1*L2)
    if cos > 1:
        return 90
    angle = np.arccos(cos) * 360/(2*np.pi)
    return angle

def count_video(cam_id, save_root):

    # load movements tipical trajs
    cam_name = cam_id
    cam_conf_json = os.path.join('cam_configs',f'{cam_id}.json')
    with open(cam_conf_json, 'r') as f:
        cam_conf = json.load(f)
    edge_mapping = cam_conf['edge_mapping']
    tipical_trajs = {edge: cam_conf[edge]['tracklets'] for edge in edge_mapping}

    #read mask image
    cam_mask = os.path.join('cam_masks',f'{cam_id}.png')
    mask = cv2.imread(cam_mask)
    h, w, c = mask.shape
    #load tracks
    tracks = {}
    tracking_json = os.path.join('runs', f'{cam_id}_tracking_data.json')
    tracks = load_tracks_from_json(tracking_json)
    
    #split tracklets
    tracks_end_in_roi, tracks_start_in_roi, tracks_too_short = check_tracks_with_roi(tracks, mask)

    trackids = sorted([k for k in tracks.keys()])
    #save count results
    os.makedirs(save_root, exist_ok=True)
    savefile = os.path.join(save_root, cam_id+'.txt')
    dst_out = open(savefile, 'w')

    #start counting
    dist_thr = 300
    angle_thr = 30
    min_length = 30
    


    cumulative_counts = defaultdict(lambda: {
                "Bicycle": 0,
                "Bus": 0,
                "Car": 0,
                "LCV": 0,
                "Three Wheeler": 0,
                "Truck": 0,
                "Two Wheeler": 0
            })
    time_intervals = defaultdict(lambda: defaultdict(lambda: {
                "Bicycle": 0,
                "Bus": 0,
                "Car": 0,
                "LCV": 0,
                "Three Wheeler": 0,
                "Truck": 0,
                "Two Wheeler": 0
            }))
    
    

    # Define class mapping
    class_mapping = {
        0: "Bicycle",
        1: "Bus",
        2: "Car",
        3: "LCV",
        4: "Three Wheeler",
        5: "Truck",
        6: "Two Wheeler"
    }

    count = 0 
    frames_per_interval = 15 * 25
    for trackid in trackids:
        if len(tracks[trackid]['tracklet']) < min_length:
            continue
        track_traj = tracks[trackid]['tracklet']
        #calc hausdorff dist with tipical trajs, assign the movement with the min dist
        all_dists_dict = {k: float('inf') for k in tipical_trajs}
        for m_id, m_t in tipical_trajs.items():
            for t in m_t:
                tmp_dist = hausdorff_distance(np.array(track_traj), np.array(t), distance='euclidean')
                if tmp_dist < all_dists_dict[m_id]:
                    all_dists_dict[m_id] = tmp_dist

        #check direction
        all_dists = sorted(all_dists_dict.items(), key=lambda k: k[1])
        # print(all_dists)
        min_idx, min_dist = None, dist_thr
        for i in range(0, len(all_dists)):
            m_id = all_dists[i][0]
            m_dist = all_dists[i][1]
            if m_dist >= dist_thr: #if min dist > dist_thr, will not assign to any movement
                break
            else:
                if is_same_direction(track_traj, tipical_trajs[m_id][0], angle_thr): #check direction
                    min_idx = m_id
                    min_dist = m_dist
                    break #if match, end
                else:
                    continue #direction not matched, find next m_id


        #remove parked vehicle detections
        track_w = tracks[trackid]['bbox'][0][3] - tracks[trackid]['bbox'][0][1]
        if abs(track_traj[-1][0] - track_traj[0][0]) < 2*track_w:
            continue

        #cam will not use shape based method because there are only 2 directions 
        direct_match_videos = ['Sty_Wll_Ldge_FIX_3', 'SBI_Bnk_JN_FIX_3','18th_Crs_BsStp_JN_FIX_2','Ayyappa_Temple_FIX_1','Devasandra_Sgnl_JN_FIX_1','Mattikere_JN_FIX_3','HP_Ptrl_Bnk_BEL_Rd_FIX_2','Kuvempu_Circle_FIX_1','Kuvempu_Circle_FIX_2','MS_Ramaiah_JN_FIX_1','Buddha_Vihara_Temple','Sundaranagar_Entrance','80ft_Road']
        if cam_name not in direct_match_videos and min_idx == None and min_dist >= dist_thr:
            continue

        if cam_name == 'Dari_Anjaneya_Temple' :
            cx_s , cy_s = track_traj[0]
            cx_e , cy_e = track_traj[-1]

            if cx_s < w*0.30 and min_idx=="BD":
                if is_same_direction(track_traj,tipical_trajs["AD"],45):
                    min_idx="AD"

            if cx_s < w*0.30 and min_idx=="BC":
                if is_same_direction(track_traj,tipical_trajs["AC"],45):
                    min_idx="AC" 
            
            if cx_s > w*0.80 and min_idx=="BD":
                if is_same_direction(track_traj,tipical_trajs["CD"],45):
                    min_idx="CD" 
                

        if cam_name == 'Mattikere_JN_HD_2' and min_idx == "AC" :
            cx_s , cy_s = track_traj[0]
            cx_e , cy_e = track_traj[-1]

            if cy_e > h*0.25 :
                continue
        
        if cam_name == 'MS_Ramaiah_JN_FIX_1' and min_idx == "AC" :
            cx_s , cy_s = track_traj[0]
            cx_e , cy_e = track_traj[-1]

            if cx_s > w*0.75 :
                continue

            if cy_e < h*0.5 :
                continue

        if cam_name == 'MS_Ramaiah_JN_FIX_2'  :
            cx_s , cy_s = track_traj[0]
            cx_e , cy_e = track_traj[-1]

            if cx_s < w*0.16 and cy_s < h*0.11 :
                continue

            if cx_e < w*0.16 and cy_e < h*0.11 :
                continue

            if min_idx == "FG":
                if is_same_direction(track_traj,tipical_trajs["FA"],45) and len(track_traj) >75 :
                    min_idx="FA"
                else:
                    continue
            
        if cam_name == 'Ramaiah_BsStp_JN_FIX_1' :
            cx_s , cy_s = track_traj[0]
            cx_e , cy_e = track_traj[-1]

            if cx_s < w*0.197 and cy_s < h*0.5 and min_idx == "AB":
                continue

            if cx_e < w*0.197 and cy_e < h*0.5 and min_idx == "BA":
                continue

            if cx_s > w*0.80 and (cy_s > h*0.13 and cy_s < h*0.37) and min_idx=="BA":
                continue

            if cx_e > w*0.80 and (cy_e > h*0.13 and cy_e < h*0.37) and min_idx=="AB":
                continue

            if cx_s < w*0.197 and cy_s < h*0.5 and cx_e > w*0.80 and (cy_e > h*0.13 and cy_e < h*0.37):
                continue

            if cx_e < w*0.197 and cy_e < h*0.5 and cx_s > w*0.80 and (cy_s > h*0.13 and cy_s < h*0.37):
                continue

        
        if cam_name == 'Ramaiah_BsStp_JN_FIX_2' :
            cx_s , cy_s = track_traj[0]
            cx_e , cy_e = track_traj[-1]


            if cx_s < w*0.25 or cx_e < w*0.25 :
                continue



        mv_idx = min_idx
        movement_id = f"{mv_idx}"

        bboxes = tracks[trackid]['bbox']
        bboxes.sort(key=lambda x: x[0])

        dst_frame = bboxes[0][0]
        last_bbox = bboxes[-1]
        if check_bbox_overlap_with_roi(last_bbox, mask) == True:
            dst_frame = last_bbox[0]
        else:
            for i in range(len(bboxes) - 2, 0, -1):
                bbox = bboxes[i]
                if check_bbox_overlap_with_roi(bbox, mask) == True:
                    dst_frame = bbox[0]
                    break
                else:
                    continue

        track_types = [int(k[5]) for k in bboxes]  # Convert to int
        track_type = max(set(track_types), key=track_types.count)

        # Map numeric class to vehicle type
        class_name = class_mapping.get(track_type, "LCV")  # Default to "LCV" if not in mapping

        end_frame = bboxes[-1][0]

        # Calculate the time interval for this track
        interval = math.floor(end_frame / frames_per_interval)

        cumulative_counts[movement_id][class_name] += 1
        time_intervals[str(interval)][movement_id][class_name] += 1
        count += 1

        # Debug: print information about each processed track
        print(f"Processed track {trackid}: movement {movement_id}, class {class_name} (numeric: {track_type})")

    
    
    # Convert defaultdict to regular dict for JSON serialization
    cumulative_counts = dict(cumulative_counts)
    time_intervals = {k: dict(v) for k, v in time_intervals.items()}

    with open(save_root+f"/{cam_id}_"+'cumulative_counts.json', 'w') as f:
        json.dump({f"{cam_id}": {"Cumulative Counts": cumulative_counts}}, f, indent=2)

    # Save time intervals
    with open(save_root+f"/{cam_id}_"+'time_intervals.json', 'w') as f:
        json.dump({f"{cam_id}": {"Time Intervals": time_intervals}}, f, indent=2)
    # Debug: print total number of processed tracks
    print(f"Total processed tracks: {count}")

    # # Save the output as JSON
    # os.makedirs(save_root, exist_ok=True)
    # save_file = os.path.join(save_root, f"{cam_id}_counts.json")
    # with open(save_file, 'w') as f:
    #     json.dump(output, f, indent=4)

    print('Vehicle counting done. Results saved in JSON format.')




