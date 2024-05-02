# cap6635_advanced-ai

A bit of a pain in the ass, in that OpenAI gym won't install "other" in anything other than Python 3.11, NOT Python 3.12 (current).
But, that being said,  
python3 -m venv ./environment  
source ./environment/bin  
pip install -r requirements.txt  

Then just:  
python dqpomdp.py -n DRQN -v ALE/Centipede-v5 -b 100000  
And:  
python dqpomdp.py -n DQN -v ALE/Centipede-v5 -b 100000  
