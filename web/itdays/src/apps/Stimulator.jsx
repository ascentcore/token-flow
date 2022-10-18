import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';

import PlayIcon from '/public/play.png';
import PauseIcon from '/public/pause.png';
import NextWordIcon from '/public/next_word.png';

export default (props) => {
  const { stimulate } = props;
  const [selectedContexts, setSelectedContexts] = useState([]);
  const [contexts, setContexts] = useState([]);
  const [play, setPlay] = useState(false);
  const [words, setWords] = useState([]);
  const [index, setIndex] = useState(-1);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    axios.get(`http://localhost:8081/contexts`).then((response) => {
      setContexts(response.data);
      setSelectedContexts(response.data);
    });
    breakSentence(
      'Hello , do you follow tennis ? Not really . I just know each player uses a strung racket and they hit a hollow ball . Did you know Polo shirts were originally invented for tennis ?'.toLowerCase()
    );
  }, []);

  function breakSentence(sentence) {
    setWords(sentence.split(' '));
    setIndex(-1);
  }

  function nextWord() {
    if (index < words.length - 1) {
      selectedContexts.forEach((context) => {
        stimulate(context, words[index + 1]);
      });
      setIndex(index + 1);
    } else {
      setIndex(-1);
    }
  }

  useEffect(() => {
    if (play) {
      nextWord();
    }
  }, [tick]);

  useEffect(() => {
    const intervalId = setInterval(() => {
      setTick((prev) => Math.random());
    }, 1000);

    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="stimulator-body">
      <div className="words">
        {words.map((word, i) => (
          <span key={i} className={i === index ? 'word active' : 'word'}>
            {word}
          </span>
        ))}
      </div>
      <div>
        {!play && (
          <img
            src={PlayIcon}
            onClick={() => {
              setPlay(true);
            }}
          />
        )}
        {play && (
          <img
            src={PauseIcon}
            onClick={() => {
              setPlay(false);
            }}
          />
        )}
        <img src={NextWordIcon} onClick={nextWord} />
      </div>
    </div>
  );
};
