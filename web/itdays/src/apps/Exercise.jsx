import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
// import 'react-tabs/style/react-tabs.css';
import fox1 from '/public/fox1.png';
import fox2 from '/public/over-lazy-dog.png';
import fox3 from '/public/repetition.png';
import fox4 from '/public/jumps-over-fence.png';

export default () => (
  <Tabs>
    <TabList>
      <Tab>Question</Tab>
      <Tab>Answer</Tab>
      <Tab>Why</Tab>
      <Tab>Context</Tab>
    </TabList>

    <TabPanel>
      <div className="image-slide">
        <div className="image-container">
          <img src={fox1} alt="Logo" />
        </div>
        <h3>The quick brown fox jumps over the ...</h3>
      </div>
    </TabPanel>
    <TabPanel>
      <div className="image-slide">
        <div className="image-container">
          <img src={fox2} alt="Logo" />
        </div>
        <h3>The quick brown fox jumps over the lazy dog</h3>
      </div>
    </TabPanel>
    <TabPanel>
      <div className="image-slide">
        <div className="image-container" style={{marginTop: 20}}>
          <img src={fox3} alt="Logo" />
        </div>
      </div>
    </TabPanel>
    <TabPanel>
      <div className="image-slide">
        <div className="image-container" style={{marginTop: 20}}>
          <img src={fox4} alt="Logo" />
        </div>
      </div>
    </TabPanel>
  </Tabs>
);
