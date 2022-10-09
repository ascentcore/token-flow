import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';
import { registerListener, triggerEvent, unregisterListener } from '../events';

export default (props) => {
  const { stimulate } = props;
  
  const [state, setState] = useState(0);
  const [data, setData] = useState([]);

  
  useEffect(() => {
    axios.get(`http://localhost:8081/vocabulary`).then((response) => {
      setData(response.data.reverse());
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
      <table cellSpacing={0}>
        <thead>
          <tr>
            <th>Token</th>
          </tr>
        </thead>
        <tbody>
          {data.map((token) => (
            <tr key={token} onClick={() => stimulate(null, token)}>
              <td>{token}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
