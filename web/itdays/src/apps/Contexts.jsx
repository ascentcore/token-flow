import axios from 'axios';
import { useState } from 'react';
import { useEffect } from 'react';
import Icon from '../components/icon';
import ContextIcon from '/public/context.png';

export default (props) => {
  const [contexts, setContexts] = useState([]);

  useEffect(() => {
    axios
      .get('http://localhost:8081/contexts')
      .then((response) => {
        const { data } = response;
        setContexts(data);
      })
      .catch((err) => {
        console.log(err);
      });
  }, []);

 
  return (
    <div className="folder">
      {contexts.map((context) => (
        <Icon
          key={context}
          name={context}
          icon={ContextIcon}
          onClick={props.openContext(context)}
        ></Icon>
      ))}
    </div>
  );
};
