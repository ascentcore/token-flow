import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
// import 'react-tabs/style/react-tabs.css';
import swdev from '/public/swdev.png';
import cto from '/public/cto.png';
import research from '/public/lead-researcher.png';
import mu from '/public/mu.png';

export default () => (
  <Tabs>
    <TabList>
      <Tab>Question</Tab>
      <Tab>Answer</Tab>
    </TabList>

    <TabPanel>
      <div class="image-slide">
        <div class="image-container">
          <img src={swdev} alt="Logo" />
        </div>
        <h3>The quick brown fox jumps over the ...</h3>
      </div>
    </TabPanel>
    <TabPanel>
      <div class="image-slide">
        <div class="image-container">
          <img src={cto} alt="Logo" />
        </div>
        <h3>The quick brown fox jumps over the fence</h3>
      </div>
    </TabPanel>
  </Tabs>
);
