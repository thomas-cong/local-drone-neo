# Project Instructions

## For Middleware 

I am building an app that essentially provides wall hacks using a flyby robotics drone that has an onboard jetson orin nx with a CSI camera and a meta quest 3 headset. i am building the middleware fastapi server that receives annotated image frames, json of segmentation/object detection metadata and sends information to the quest 3 about these received frames so it can produce the correct visual output. additionally, the quest 3 will be sending its mic captured audio stream to this fastapi server for transcription/reasoning. lets take this step by step. 