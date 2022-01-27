cap = cv2.VideoCapture(r'C:\Users_test.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

history=[]

video = cv2.VideoWriter('video2_test.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, (800, 600))
df = pd.DataFrame()

i=0

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    #cv2.putText(cv2.resize(image_np_with_detections, (800, 600)), text="GHD.AI")  
    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    cv2.imwrite( str(i) + '_infer.png', cv2.resize(image_np_with_detections, (800, 600)))
    video.write(cv2.imread(str(i) + '_infer.png'))
    

    
    
    #
    word = category_index[detections['detection_classes'][np.argmax(detections['detection_scores'])]+1]['name']
    
    #if(np.argmax(detections['detection_scores'])>0):
    #    word1 = np.max(detections['detection_scores'])
    #    history.append(word1)
    
    #for index, (key, value) in enumerate(detections.items()):
    #    #time.sleep(5)
    #    if index == 1 :
    #        print(key, '::', value)  
    
   # time.sleep(2)
    #print(history)
    #word1= [n for n in detections['detection_scores'] if n >= 0.6]
    
    
    
    word1=np.where(detections['detection_scores'] > 0.67)[0]
    
    print(detections['detection_classes'][word1])
    print(detections['detection_scores'][word1])
    
    
    #df = pd.DataFrame(detections['detection_classes'][word1])
    
    data1 = pd.DataFrame({"class": detections['detection_classes'][word1]})
    
    
    data2 = pd.DataFrame({"scores": detections['detection_scores'][word1]})
    
    
    
    data= pd.concat([data1, data2], axis=1)
    
    data['frame']=i
    
    df = df.append(data)
    
    i= i+1
    if i>250:
        break
    
    #print(detections['detection_scores'])
    #print(word1)
    
   # for item in detections:
   #     print("Key : {} , Value : {}".format(item,detections[item]))
             
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()
