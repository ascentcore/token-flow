import axios from 'axios';
import { useRef } from 'react';
import { useState } from 'react';
import { useEffect } from 'react';
import { registerListener, triggerEvent, unregisterListener } from '../events';

export default (props) => {
  const { graph } = props;

  const canvasRef = useRef(null);
  const [word, setWord] = useState(null)
  const [vocab, setVocab] = useState(null);
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get(`http://localhost:8081/cache`).then((response) => {
      const { data } = response;
      setVocab(Object.keys(data));
      setData(Object.values(data));

      Object.values(data).forEach((row) => {
        const [x, y] = row;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        context.fillStyle = `rgba(255, 125, 125, 1`;
        context.fillRect(x * 500, y * 500, 2, 2);
        console.log(x, y);
      });
    });
  }, []);

  function distanceBetweenPoints(p1, p2) {
    return Math.abs(
      Math.sqrt(
        (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])
      )
    );
  }

  const click = (event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (event.clientX - rect.left) / 500;
    const y = (event.clientY - rect.top) / 500;
    const distances = vocab.map((word, index) => ({
      word,
      distance: distanceBetweenPoints([x, y], data[index]),
    })).sort( (a,b) => a.distance - b.distance)
    setWord(distances[0].word)
  };

  return (
    <div className="vocabulary-body">
      <div>Word: {word}</div>
      <canvas ref={canvasRef} width={500} height={500} onClick={click} />
    </div>
  );
};
