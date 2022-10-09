import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';
import { registerListener, triggerEvent, unregisterListener } from '../events';

export default (props) => {
  const { graph, context, stimulate } = props;

  const [data, setData] = useState([]);
  const [showTitle, setShowTitle] = useState(true);
  const [error, setError] = useState(false);
  const [showValues, setShowValues] = useState(true);
  const [stimuli, setStimuli] = useState({});
  const [header, setHeader] = useState([]);

  useEffect(() => {
    if (graph) {
      const localData = [];
      const { nodes, links } = graph;
      const header = nodes.map((item) => item.id);

      if (header.length < 100) {
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
        setHeader(header);
      } else {
        setError(true);
      }
    }
  }, [graph]);

  const getColorFor = (token) => {
    return `rgba(255, 125, 125, ${stimuli[token]})`;
  };

  const getColorForValue = (value) => {
    return `rgba(125, 125, 255, ${value})`;
  };

  return (
    <div className="vocabulary-body">
      {error && <div className="error">Too many nodes to display</div>}
      {!error && (
        <table cellSpacing={0}>
          <thead>
            <tr>
              <th></th>
              {header.map((token, rowIndex) => (
                <th
                  key={token}
                  style={{ backgroundColor: getColorFor(token), padding: 1 }}
                >
                  {showTitle && token}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`}>
                <td style={{ backgroundColor: getColorFor(header[rowIndex]) }}>
                  {showTitle && header[rowIndex]}
                </td>
                {row.map((value, colIndex) => (
                  <td
                    key={`${colIndex}-${rowIndex}`}
                    style={
                      rowIndex === colIndex
                        ? {
                            backgroundColor: getColorFor(header[rowIndex]),
                          }
                        : {
                            backgroundColor: getColorForValue(value),
                          }
                    }
                  >
                    {showValues && Math.floor(value * 100) / 100}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};
