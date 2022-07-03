I changed the environment such that it would reset to a previous "save" maximum 200 steps before losing a life - "replaying" the failure for 5 times. And then adding that savegame to a list from which it would sometimes draw a save to start from.
90% probability a new game would be started. 10% an old save would be replayed

I added a callcback monitoring the metrics for how many games are replayed, how many saved, how many started fresh.

In training it became evident, that the agent would get stuck in the evaluation environment - the length of eval episodes would increase sharply  at 12m training steps.
I have not understood why it is stuck, but it was resolved with turning on again the EpisodicLifeEnv, which we had turned of beacuse of the issue with "trying to step environment that needs reset".
