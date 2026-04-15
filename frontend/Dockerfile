FROM node:20-alpine AS build

WORKDIR /app

COPY frontend/package*.json ./

ARG VITE_BACKEND_URL=http://127.0.0.1:8000
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}

RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
