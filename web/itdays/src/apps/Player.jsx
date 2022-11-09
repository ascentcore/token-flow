import { useState } from 'react';
import { useEffect } from 'react';

import WinampSkin from '/public/winamp.png';
import { triggerEvent } from '../events';
const chats = [
  {
    "message": "do you like dance?",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes  I do. Did you know Bruce Lee was a cha cha dancer?",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes he even won a hardcore cha cha championship in 1958",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yeah. Did you know Tupac was a ballet dancer?",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes and he even was in the production of the nutcracker",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yeah. Ballet dancer go through 4 pairs of shoes a week",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes that is a lot of shoes and also a lot of money",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yeah true. Did you know babies are really good at dancing?",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes and they smile more when they hit the beat",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yeah they are much smarter than we give them credit for",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "True Did you know Jackson had a patent on a dancing device?",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes it helped him smooth out his dance moves",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Nice. Do you like Shakespeare?",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS2"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes I do. Do you know that he popularized many phrases",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS2"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes like good riddance, in my heart of hearts and such",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS2"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes and then he also invented names like Jessica, Olivia and Miranda",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS2"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes. And for his works you have to use old english for it to make sense",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS2"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes otherwise the rhymes and puns do not seem to work out",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS2"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yes. He lived at the same time as Pocahontas too",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS2"],
    "turn_rating": "Excellent"
  },
  {
    "message": "I wonder if they met how that would go from there",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS2"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Yeah interesting point. Nice chat",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["Personal Knowledge"],
    "turn_rating": "Excellent"
  },
  {
    "message": "Do you like tennis? I just love to play the sport.",
    "agent": "agent_1",
    "sentiment": "Happy",
    "knowledge_source": ["Personal Knowledge"],
    "turn_rating": "Good"
  },
  {
    "message": "I've played it a couple time in high school gym class, I wasn't that good",
    "agent": "agent_2",
    "sentiment": "Neutral",
    "knowledge_source": ["FS2"],
    "turn_rating": "Poor"
  },
  {
    "message": "i love it! Tennis is a racket sport that can be played individually against a single opponent or in teams",
    "agent": "agent_1",
    "sentiment": "Happy",
    "knowledge_source": ["FS1"],
    "turn_rating": "Excellent"
  },
  {
    "message": "I watch it sometimes when Serena Williams is playing",
    "agent": "agent_2",
    "sentiment": "Neutral",
    "knowledge_source": ["FS2"],
    "turn_rating": "Passable"
  },
  {
    "message": "yeah i like to watch her play. Polo shirts were originally invented for tennis by famous player rene \"the crocodile\" lacoste. i love their colgne for men",
    "agent": "agent_1",
    "sentiment": "Neutral",
    "knowledge_source": ["FS1"],
    "turn_rating": "Good"
  },
  {
    "message": "Intresting, the longest match played in a polo shirt was 22 hours",
    "agent": "agent_2",
    "sentiment": "Neutral",
    "knowledge_source": ["FS1"],
    "turn_rating": "Good"
  },
  {
    "message": "yeah if i recall it went over three days or so",
    "agent": "agent_1",
    "sentiment": "Neutral",
    "knowledge_source": ["FS1"],
    "turn_rating": "Good"
  },
  {
    "message": "That's a long time, I don't think I've ever done anything for 3 days straight",
    "agent": "agent_2",
    "sentiment": "Surprised",
    "knowledge_source": ["FS1"],
    "turn_rating": "Passable"
  },
  {
    "message": "im not sure if its true but Us open quarterfinalist gael monfils beat the \"roger federer of paddle tennis\" after learning how to play just days earlier.",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS1"],
    "turn_rating": "Good"
  },
  {
    "message": "Man that's crazy, almost as crazy as using a helicopter to dry grass tennis courts",
    "agent": "agent_2",
    "sentiment": "Surprised",
    "knowledge_source": ["FS1"],
    "turn_rating": "Good"
  },
  {
    "message": "yeah i have heard that as well and though it sounded really dangerous. i wonder if their were people in the crowd?",
    "agent": "agent_1",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["Personal Knowledge"],
    "turn_rating": "Good"
  },
  {
    "message": "I hope not, if I was there I'd be cold",
    "agent": "agent_2",
    "sentiment": "Neutral",
    "knowledge_source": ["Personal Knowledge"],
    "turn_rating": "Poor"
  },
  {
    "message": "yeah. In 1998, serena and venus williams said they could beat any man ranked 200 or worse in a game of tennis. they lost when challenged twice lol",
    "agent": "agent_1",
    "sentiment": "Neutral",
    "knowledge_source": ["FS2"],
    "turn_rating": "Good"
  },
  {
    "message": "lol, well she did win the australian open when she was ranked 95th",
    "agent": "agent_2",
    "sentiment": "Neutral",
    "knowledge_source": ["FS2"],
    "turn_rating": "Good"
  },
  {
    "message": "i have heard Serena williams is a co-owner of the miami dolphins",
    "agent": "agent_1",
    "sentiment": "Neutral",
    "knowledge_source": ["FS2"],
    "turn_rating": "Good"
  },
  {
    "message": "I didn't know that, she speaks french and italian so I wonder if she owns any sports teams in Europe",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["FS2"],
    "turn_rating": "Passable"
  },
  {
    "message": "not sure but In january 2007, serena williams was ranked 95th in the world in tennis",
    "agent": "agent_1",
    "sentiment": "Neutral",
    "knowledge_source": ["FS2"],
    "turn_rating": "Good"
  },
  {
    "message": "I never knew she was ever ranked that low",
    "agent": "agent_2",
    "sentiment": "Neutral",
    "knowledge_source": ["Personal Knowledge"],
    "turn_rating": "Not Good"
  },
  {
    "message": "well she reached the No. 1 ranking for the first time on July 8, 2002 but has since fallen",
    "agent": "agent_1",
    "sentiment": "Neutral",
    "knowledge_source": ["FS2"],
    "turn_rating": "Good"
  },
  {
    "message": "Wow, shes so good though, she won a grand slam while pregnant",
    "agent": "agent_2",
    "sentiment": "Surprised",
    "knowledge_source": ["FS2"],
    "turn_rating": "Passable"
  },
  {
    "message": "I guess in my opinion her and her sister are kinda overated.",
    "agent": "agent_1",
    "sentiment": "Neutral",
    "knowledge_source": ["Personal Knowledge"],
    "turn_rating": "Good"
  },
  {
    "message": "I don't know, maybe",
    "agent": "agent_2",
    "sentiment": "Curious to dive deeper",
    "knowledge_source": ["Personal Knowledge"],
    "turn_rating": "Poor"
  }
]

export default (props) => {

  const { stimulate } = props;
  const [ index, setIndex ] = useState(0);


  useEffect(() => {
    if (index < chats.length-1) {
      setTimeout( () => {
        triggerEvent('talk', `${chats[index].agent}: ${chats[index].message}`)
        stimulate(chats[index+1].agent, chats[index].message).then( () => {
          console.log('stimulated')
          setIndex(index => index+1);
        })
      }, 5000)
      
    }

  }, [index])

  return (
    <div className="stimulator-body">
      <img src={WinampSkin}/>
    </div>
  )
};
