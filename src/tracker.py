import cv2

# OpenCV has a bunch of object tracking algorithms. We list them here.
type_of_trackers = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN',
                    'MOSSE', 'CSRT']

# CSRT is accurate but slow. You can try others and see what results you get.

def generate_tracker(type_of_tracker):
    """
    Create object tracker.

    :param type_of_tracker string: OpenCV tracking algorithm
    """
    if type_of_tracker == type_of_trackers[0]:
        tracker = cv2.TrackerBoosting_create()
    elif type_of_tracker == type_of_trackers[1]:
        tracker = cv2.TrackerMIL_create()
    elif type_of_tracker == type_of_trackers[2]:
        tracker = cv2.TrackerKCF_create()
    elif type_of_tracker == type_of_trackers[3]:
        tracker = cv2.TrackerTLD_create()
    elif type_of_tracker == type_of_trackers[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif type_of_tracker == type_of_trackers[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif type_of_tracker == type_of_trackers[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif type_of_tracker == type_of_trackers[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('The name of the tracker is incorrect')
        print('Here are the possible trackers:')
        for track_type in type_of_trackers:
            print(track_type)
    return tracker
