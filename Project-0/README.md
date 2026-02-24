# Project-0
Spot to refine the process of starting a project. Right now it contains 2 main scripts. "0-system_setup.sh" is used to get a Linux system up to work with models. Canine health detection is the project that gets set up from the other script. It is a single script that gets from setup to running a server with UI for audio detection.

# Models Uploaded at 
https://drive.google.com/drive/folders/1hR3dpWm2BQN8gsHzzWSucL9ystydvYeg?usp=sharing

# TODO
Anyone is free to add thoughts here...

Before we get to VM we still need to create the ground zero script which will setup the network and all those things which I believe already exists in EFBackend project. Please move this to wherever needed and then delete/rename this project.

We also need to create the layer that decides where to route a request. Which model gets triggered by which type of request. All these thoughts will decide into the architecture of where the networking/setup/orchestration/monitoring falls into play. I would say design a set of systems that can have each individual service/app/purpose built with one script each. This can then wake up as a service, call home to some orchestrator which can then route traffic to this service based on the request. This will allow for the services to individually scale based on bursts of intent (i.e. 400,000 videos being pumped in). Looking forward to a script/scale driven thought process. I like what Soumith mentioned on his page. Laziness is a great motivator.

Thank you :)
