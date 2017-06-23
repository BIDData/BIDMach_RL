# BIDMach_RL
BIDMach project for state-of-the-art RL algorithms. Currently includes N-step DQN and A3C (arguably its A2C since it uses a GPU and is only partly asynchronous). 

To build and run using Apache Maven, do:
<pre>
git clone http://github.com/BIDData/BIDMach_RL
cd BIDMach_RL
mvn clean install
</pre>

You run Atari games you will need to put the appropriate ROMs in BIDMach_RL/roms. 

To run the scripts, do:
<pre> 
cd scripts
../bidmach &lt;ScriptName&gt;
</pre>
