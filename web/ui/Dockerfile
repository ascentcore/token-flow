FROM node:18

RUN apt-get update

WORKDIR /app

COPY package*.json ./

RUN npm install --legacy-peer-deps 

COPY . .

CMD ["npm", "start"]
