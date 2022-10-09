import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';
import { registerListener, triggerEvent, unregisterListener } from '../events';

export default (props) => {
  const { graph, context, stimulate } = props;

  const [data, setData] = useState([]);

  useEffect(() => {
    const data = graph.nodes.map((item) => [item.id, item.s]).sort( (a,b) => b[1] - a[1]);
    setData(data)
  }, [graph]);

  return (
    <div className="vocabulary-body">
      <table cellSpacing={0}>
        <thead>
          <tr>
            <th>Token</th>
            <th>Stimuli</th>
          </tr>
        </thead>
        <tbody>
          {data.map((node) => (
            <tr key={node[0]} onClick={() => stimulate(context, node[0])} style={{backgroundColor: `rgba(255,125,125,${node[1]})`}}>
              <td>{node[0]}</td>
              <td>{Math.floor(node[1] * 100) / 100}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
