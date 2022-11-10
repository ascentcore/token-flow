import axios from 'axios';
import { useRef } from 'react';
import { useState } from 'react';
import { useEffect } from 'react';
import { registerListener, triggerEvent, unregisterListener } from '../events';

export default (props) => {
  const { graph } = props;

  const canvasRef = useRef(null);

  const [data, setData] = useState([]);
  const [multiplier, setMultiplier] = useState(4);
  const [stimuli, setStimuli] = useState({});

  useEffect(() => {
    if (graph) {
      const localData = [];
      const { nodes, links } = graph;
      const header = nodes.map((item) => item.id);

      // if (header.length < 50) {
      //   setMultiplier(8);
      // } else if (header.length < 100) {
      //   setMultiplier(6);
      // } else if (header.length < 500) {
      //   setMultiplier(4);
      // } else {
      //   setMultiplier(2);
      // }

      // setMultiplier(4)

      for (let i = 0; i < header.length; i++) {
        localData.push(new Array(header.length).fill(0));
      }

      links.forEach((link) => {
        const { source, target, weight } = link;

        const row = header.indexOf(source);
        const column = header.indexOf(target);

        localData[row][column] = weight;
      });

      setStimuli(
        nodes.reduce((memo, item) => {
          memo[item.id] = item.s;
          return memo;
        }, {})
      );
      setData(localData);
    }
  }, [graph]);

  useEffect(() => {
    console.log(data, stimuli);
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const stim = Object.values(stimuli);

    for (let i = 0; i < data.length; i++) {
      context.fillStyle = `rgba(255, 125, 125, ${stim[i]})`;
      context.fillRect(i , 0, multiplier, multiplier);

      context.fillRect(0, i , multiplier, multiplier);
    }

    for (let i = 0; i < data.length; i++) {
      for (let j = 0; j < data[i].length; j++) {
        context.fillStyle = `rgba(125, 125, 255, ${data[i][j]})`;
        context.fillRect(
          (i + 1) ,
          (j + 1) ,
          multiplier,
          multiplier
        );
      }
    }
  }, [data]);

  return (
    <div className="vocabulary-body">
      <canvas
        ref={canvasRef}
        width={(data.length + 2) }
        height={(data.length + 2)}
        style={{width: 400}}
      />
    </div>
  );
};
