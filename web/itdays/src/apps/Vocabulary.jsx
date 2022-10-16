import axios from 'axios';
import { autoType } from 'd3';
import { useState } from 'react';
import { useEffect } from 'react';
import { registerListener, triggerEvent, unregisterListener } from '../events';

export default (props) => {
  const { stimulate } = props;

  const [state, setState] = useState(0);
  const [data, setData] = useState([]);
  const [query, setQuery] = useState('');

  useEffect(() => {
    axios.get(`http://localhost:8081/vocabulary`).then((response) => {
      setData(
        response.data
          .sort((a, b) => a.localeCompare(b))
      );
      triggerEvent('vocabulary', response.data.length);
    });
  }, [state]);

  useEffect(() => {
    const callback = (data) => {
      setState(data);
    };
    registerListener('global', callback);
    return () => {
      console.log('Unmounting ... ');
      unregisterListener(callback);
    };
  }, []);

  return (
    <div className="vocabulary-body">
      <div>Filter:</div>
      <input
        style={{ width: '100%' }}
        type="text"
        placeholder="Search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="vocabulary-table">
        <table cellSpacing={0}>
          <thead>
            <tr>
              <th>Token</th>
            </tr>
          </thead>
          <tbody>
            {data.map(
              (token) =>
                token.indexOf(query) === 0 && (
                  <tr key={token} onClick={() => stimulate(null, token)}>
                    <td>{token}</td>
                  </tr>
                )
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
